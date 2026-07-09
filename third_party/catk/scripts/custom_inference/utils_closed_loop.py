import argparse
import hashlib
import math
import os
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from waymo_open_dataset.protos import scenario_pb2

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# Make sure local imports work when launched from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(_PROJECT_ROOT))

os.environ.setdefault("CATK_SKIP_TF", "1")

from src.smart.model.smart import SMART as SMARTBase
from src.smart.model.smart_ctrl import SMART as SMARTCtrl
from src.smart.utils import compute_batch_boundaries, truncate_batch_by_boundary
from src.utils.vis_waymo import get_map_features
from utils_closed_loop_latent import align_latents_to_agents, load_latent_payload
from utils_closed_loop_sketch import (
    build_custom_sketch,
    build_gt_sketch,
    build_right_turn_sketch,
    extract_controlled_agent_indices,
    save_custom_sketch_points,
)
from utils_closed_loop_text import (
    _build_control_text_for_batch,
    _build_control_text_from_custom,
    _load_text_mapping,
    _parse_custom_text,
)

N_SIM_AGENTS = 32
VIEW_RANGE = 75
COORD_LIMIT = 100_000


LANE_STATE_MAP = {
    0: 0,  # UNKNOWN
    1: 1,  # STOP
    2: 2,  # CAUTION
    3: 3,  # GO
    4: 4,  # FLASHING_STOP
    5: 4,  # FLASHING_CAUTION
}


class ScenarioDataset(Dataset):
    def __init__(self, data_paths: List[Path]) -> None:
        super().__init__()
        self.data_paths = data_paths

    def len(self) -> int:
        return len(self.data_paths)

    def get(self, idx: int):
        with open(self.data_paths[idx], "rb") as f:
            data = pickle.load(f)
        return HeteroData(data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_catk_pkls(catk_dir: Path) -> List[Path]:
    return sorted(catk_dir.glob("*.pkl"))

def list_scenario_ids_from_dir(source_dir: Path) -> List[str]:
    if not source_dir.exists():
        raise RuntimeError(f"Scenario id directory not found: {source_dir}")
    scenario_ids: List[str] = []
    seen = set()
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        if "scenario_" not in stem:
            continue
        tail = stem.split("scenario_")[-1]
        sid = tail.split("_")[0]
        if sid and sid not in seen:
            scenario_ids.append(sid)
            seen.add(sid)
    return scenario_ids


def select_catk_scenarios(
    catk_dir: Path,
    seed: int,
    max_scenarios: int,
    scenario_ids: List[str] | None = None,
) -> List[Path]:
    if scenario_ids:
        paths = []
        missing = []
        for sid in scenario_ids:
            sid = sid.replace("scenario_", "")
            pkl_path = catk_dir / f"{sid}.pkl"
            if pkl_path.exists():
                paths.append(pkl_path)
            else:
                missing.append(sid)
        if missing:
            print(
                "Warning: missing CATK scenarios: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        return paths

    pkl_files = list_catk_pkls(catk_dir)
    if max_scenarios > 0:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pkl_files))[:max_scenarios]
        pkl_files = [pkl_files[i] for i in order]
    return pkl_files


def _ensure_hydra(output_dir: Path) -> None:
    if HydraConfig.initialized():
        return

    @dataclass
    class _DummyConfig:
        hydra: HydraConf = field(default_factory=HydraConf)

    cfg = OmegaConf.structured(_DummyConfig())
    cfg.hydra.runtime.output_dir = str(output_dir)
    HydraConfig.instance().set_config(cfg)


def _clean_training_loss(cfg) -> None:
    if not hasattr(cfg, "training_loss"):
        cfg.training_loss = OmegaConf.create({})
    allowed = {
        "use_gt_raw",
        "gt_thresh_scale_length",
        "label_smoothing",
        "rollout_as_gt",
    }
    defaults = {
        "use_gt_raw": True,
        "gt_thresh_scale_length": -1.0,
        "label_smoothing": 0.1,
        "rollout_as_gt": False,
    }
    tl = cfg.training_loss
    cleaned = {k: v for k, v in tl.items() if k in allowed}
    for key, value in defaults.items():
        cleaned.setdefault(key, value)
    cfg.training_loss = OmegaConf.create(cleaned)


def load_model_from_checkpoint(
    ckpt_path: str, model_cls, device: torch.device, output_dir: Path
):
    _ensure_hydra(output_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper = ckpt.get("hyper_parameters", {})
    model_cfg = hyper.get("model_config")
    if model_cfg is None and "model" in hyper:
        model_cfg = hyper["model"].get("model_config")
    if model_cfg is None:
        raise RuntimeError("Could not find model_config in checkpoint hyper_parameters")
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)
    _clean_training_loss(model_cfg)
    model = model_cls(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def apply_rollout_boundary_if_available(model, batch, pred_dict):
    pred_traj = pred_dict.get("pred_traj_10hz")
    pred_head = pred_dict.get("pred_head_10hz")
    if pred_traj is None or pred_head is None:
        return pred_dict
    map_boundary_cfg = getattr(model, "map_boundary_cfg", {})
    if not bool(map_boundary_cfg.get("enable", False)):
        return pred_dict
    boundaries = compute_batch_boundaries(
        batch["map_save"]["traj_pos"],
        batch["pt_token"]["batch"],
        batch.num_graphs,
        margin=float(map_boundary_cfg.get("margin", 3.0)),
    )
    n_agent, n_step, _ = pred_traj.shape
    valid_mask = torch.ones(
        (n_agent, n_step), dtype=torch.bool, device=pred_traj.device
    )
    valid_mask = truncate_batch_by_boundary(
        valid_mask,
        pred_traj,
        batch["agent"]["batch"],
        boundaries,
        start_step=0,
    )
    pred_traj = pred_traj.clone()
    pred_head = pred_head.clone()
    invalid = ~valid_mask
    pred_traj[invalid] = -1.0
    pred_head[invalid] = 0.0
    pred_dict["pred_traj_10hz"] = pred_traj
    pred_dict["pred_head_10hz"] = pred_head
    pred_dict["pred_valid_10hz"] = valid_mask
    pred_z = pred_dict.get("pred_z_10hz")
    if pred_z is not None:
        pred_z = pred_z.clone()
        pred_z[invalid] = 0.0
        pred_dict["pred_z_10hz"] = pred_z
    return pred_dict


def load_waymo_scenario(tfrecord_path: Path) -> scenario_pb2.Scenario | None:
    if os.environ.get("CATK_SKIP_TF", "") == "1":
        return None
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"Warning: TensorFlow import failed; skip TFRecord map/traffic lights ({exc})")
        return None

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    scenario = scenario_pb2.Scenario()
    try:
        for data in tf.data.TFRecordDataset([tfrecord_path.as_posix()], compression_type=""):
            scenario.ParseFromString(bytes(data.numpy()))
            break
    except Exception as exc:
        print(f"Warning: failed to read TFRecord {tfrecord_path.name}: {exc}")
        return None
    return scenario


def build_polylines(scenario: scenario_pb2.Scenario) -> List[np.ndarray]:
    mp_xyz, _, _ = get_map_features(scenario.map_features)
    polylines = []
    for line in mp_xyz:
        arr = np.asarray(line)
        if arr.size == 0:
            continue
        polylines.append(arr[:, :2])
    return polylines


def build_polylines_from_map_save(map_save) -> List[np.ndarray]:
    try:
        traj_pos = map_save["traj_pos"]
    except Exception:
        return []

    traj_pos_np = _to_numpy(traj_pos)
    if traj_pos_np.ndim != 3 or traj_pos_np.shape[-1] < 2:
        return []

    polylines = []
    for line in traj_pos_np:
        if not np.isfinite(line).all():
            continue
        # Drop dummy polylines (preprocess uses ~6e4 sentinel for empty maps).
        if np.max(np.abs(line)) > 1.0e4:
            continue
        if np.allclose(line, 0.0):
            continue
        polylines.append(line[:, :2])
    return polylines


def build_traffic_light_points(
    scenario: scenario_pb2.Scenario, step_idx: int
) -> np.ndarray:
    if not scenario.dynamic_map_states:
        return np.zeros((0, 3), dtype=np.float32)
    step_idx = min(step_idx, len(scenario.dynamic_map_states) - 1)
    points = []
    for lane_state in scenario.dynamic_map_states[step_idx].lane_states:
        state = LANE_STATE_MAP.get(lane_state.state, 0)
        stop = lane_state.stop_point
        points.append([stop.x, stop.y, state])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_agents_history_future(
    position: torch.Tensor | np.ndarray,
    heading: torch.Tensor | np.ndarray,
    velocity: torch.Tensor | np.ndarray,
    shape: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray,
    num_historical_steps: int,
    num_future_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = _to_numpy(position)[..., :2]
    head = _to_numpy(heading)
    vel = _to_numpy(velocity)
    valid = _to_numpy(valid_mask).astype(bool)
    dims = _to_numpy(shape)

    num_agents = pos.shape[0]
    hist_len = min(num_historical_steps, pos.shape[1])
    fut_len = min(num_future_steps, max(0, pos.shape[1] - hist_len))

    agents_history = np.zeros((num_agents, hist_len, 8), dtype=np.float32)
    agents_future = np.zeros((num_agents, fut_len, 5), dtype=np.float32)

    agents_history[:, :, 0:2] = pos[:, :hist_len]
    agents_history[:, :, 2] = head[:, :hist_len]
    agents_history[:, :, 3:5] = vel[:, :hist_len]

    length = dims[:, 0:1]
    width = dims[:, 1:2] if dims.shape[1] > 1 else dims[:, 0:1]
    height = dims[:, 2:3] if dims.shape[1] > 2 else np.ones_like(length)
    agents_history[:, :, 5] = length
    agents_history[:, :, 6] = width
    agents_history[:, :, 7] = height

    hist_valid = valid[:, :hist_len]
    agents_history[~hist_valid] = 0.0

    if fut_len > 0:
        start = hist_len
        end = hist_len + fut_len
        agents_future[:, :, 0:2] = pos[:, start:end]
        agents_future[:, :, 2] = head[:, start:end]
        agents_future[:, :, 3:5] = vel[:, start:end]
        fut_valid = valid[:, start:end]
        agents_future[~fut_valid] = 0.0

    return agents_history, agents_future


def build_train_control_mask(
    batch: HeteroData,
    max_num: int,
    seed: Optional[int] = None,
    scenario_id: Optional[str] = None,
    step_current: int = 10,
    dist_limit: float = 100.0,
    valid_radius: float = 150.0,
    min_future_steps: int = 5,
) -> Optional[torch.Tensor]:
    pos = batch["agent"]["position"]
    role = batch["agent"]["role"]
    valid_mask = batch["agent"]["valid_mask"]

    if pos.dim() != 3 or valid_mask.dim() != 2:
        return None

    role = role.to(torch.bool)
    av_candidates = torch.where(role[:, 0])[0]
    av_index = int(av_candidates[0]) if av_candidates.numel() > 0 else 0

    distance = torch.norm(pos - pos[av_index], dim=-1)
    valid_mask = valid_mask.to(torch.bool) & (distance < valid_radius)

    step_idx = min(step_current, distance.shape[1] - 1)
    if step_idx + 1 < valid_mask.shape[1]:
        future_valid = valid_mask[:, step_idx + 1 :].sum(-1) >= min_future_steps
    else:
        future_valid = torch.zeros(
            valid_mask.shape[0], device=valid_mask.device, dtype=torch.bool
        )

    role_train_mask = role.any(dim=-1)
    extra_train_mask = (distance[:, step_idx] < dist_limit) & future_valid
    train_mask = extra_train_mask | role_train_mask
    if train_mask.sum().item() <= max_num:
        return train_mask

    role_count = int(role_train_mask.sum().item())
    if role_count >= max_num:
        return role_train_mask

    candidates = extra_train_mask & ~role_train_mask
    candidate_idx = torch.where(candidates)[0].tolist()
    if not candidate_idx:
        return role_train_mask

    if seed is None:
        seed = 0
    if scenario_id:
        scenario_key = int(hashlib.md5(scenario_id.encode("utf-8")).hexdigest(), 16)
        seed = (seed + scenario_key) % (2**31 - 1)

    rng = np.random.default_rng(seed)
    select_count = min(max_num - role_count, len(candidate_idx))
    picked = rng.choice(candidate_idx, size=select_count, replace=False)

    selected = role_train_mask.clone()
    selected[picked] = True
    return selected


def build_waymo_target_control_mask(batch: HeteroData) -> Optional[torch.Tensor]:
    role = batch["agent"].get("role", None)
    if role is None or role.dim() != 2 or role.shape[1] < 3:
        return None
    return role[:, 2].to(torch.bool)


def build_wosac_rollout(
    pred_traj: np.ndarray,
    pred_head: np.ndarray,
    pred_z: Optional[np.ndarray] = None,
    pred_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if pred_traj.ndim != 3:
        raise ValueError(f"pred_traj must be [A,T,2], got {pred_traj.shape}")
    if pred_head.ndim != 2:
        raise ValueError(f"pred_head must be [A,T], got {pred_head.shape}")
    num_agents, num_steps, _ = pred_traj.shape
    rollout = np.zeros((num_agents, num_steps, 4), dtype=np.float32)
    rollout[:, :, 0:2] = pred_traj.astype(np.float32)
    if pred_z is not None:
        rollout[:, :, 2] = pred_z.astype(np.float32)
    rollout[:, :, 3] = pred_head.astype(np.float32)

    if pred_valid is not None:
        valid_mask = pred_valid.astype(bool)
    else:
        is_zero = np.all(rollout == 0.0, axis=-1)
        is_neg = (rollout[:, :, 0] == -1.0) & (rollout[:, :, 1] == -1.0)
        invalid = is_zero | is_neg
        invalid_cum = np.maximum.accumulate(invalid.astype(np.int32), axis=-1).astype(bool)
        valid_mask = ~invalid_cum
    return rollout, valid_mask


def save_wosac_pkl(
    output_dir: Path,
    scenario_id: str,
    rollouts: Dict[str, Dict[str, np.ndarray]],
    agents_id: np.ndarray,
    sim_agent_mask: np.ndarray,
    num_steps: int,
    num_rollouts: int,
    control_sketch: Optional[np.ndarray],
    controlled_agents: Optional[List[int]],
    seed: int,
) -> None:
    payload = {
        "scenario_id": scenario_id,
        "model_rollouts": rollouts,
        "agents_id": agents_id,
        "sim_agent_mask": sim_agent_mask,
        "num_agents": len(agents_id),
        "num_rollouts": num_rollouts,
        "num_steps": num_steps,
        "horizon_steps": num_steps,
        "total_steps": num_steps,
        "num_iterations": 1,
        "sketch": control_sketch,
        "controlled_agents": controlled_agents or [],
        "seed": seed,
        "test_mode": "catk",
    }
    out_path = output_dir / f"{scenario_id}_closedloop_wosac.pkl"
    with open(out_path, "wb") as handle:
        pickle.dump(payload, handle)


def sketch_to_future(sketch_2hz: np.ndarray, target_steps: int) -> np.ndarray:
    if sketch_2hz.shape[1] == 0:
        return np.zeros((sketch_2hz.shape[0], target_steps, 5), dtype=np.float32)
    repeated = np.repeat(sketch_2hz, 5, axis=1)
    if repeated.shape[1] < target_steps:
        pad_len = target_steps - repeated.shape[1]
        pad = np.repeat(repeated[:, -1:, :], pad_len, axis=1)
        repeated = np.concatenate([repeated, pad], axis=1)
    if repeated.shape[1] > target_steps:
        repeated = repeated[:, :target_steps]

    future = np.zeros((repeated.shape[0], target_steps, 5), dtype=np.float32)
    future[:, :, 0:2] = repeated[:, :, 0:2]
    future[:, :, 2] = repeated[:, :, 2]

    invalid = np.abs(repeated[:, :, 0]) + np.abs(repeated[:, :, 1]) < 1e-6
    future[:, :, 0][invalid] = -1.0
    future[:, :, 1][invalid] = -1.0
    return future


def make_pred_traj_xyyaw(pred_traj: np.ndarray, pred_head: np.ndarray) -> np.ndarray:
    out = np.zeros((pred_traj.shape[0], pred_traj.shape[1], 3), dtype=np.float32)
    out[:, :, 0:2] = pred_traj
    out[:, :, 2] = pred_head
    return out


def create_trajectory_video(
    pkl_data: dict,
    pred_trajs: List[Tuple[str, np.ndarray]],
    save_path: str,
    scenario_name: str,
    conditioning_future: np.ndarray | None,
    controlled_agent_ids: List[int] | None,
    show_agent_ids: bool = False,
) -> None:
    original_style = plt.rcParams["axes.prop_cycle"]
    plt.style.use("default")

    palette = {
        "map": "#B0B0B0",
        "gt_future": "#F5B7B1",
        "pred_future": "#7FB3D5",
        "pred_future_alt": "#D7BDE2",
        "condition_future": "#F8C471",
        "ego_box": "#DE5959",
        "vehicle_box": "#87B3E6",
        "ped_box": "#BEA9F5",
        "cyc_box": "#2ECC71",
        "other_box": "#9E9E9E",
        "controlled_box": "#FF9F1A",
    }

    agents_history = pkl_data["agents_history"]
    agents_future = pkl_data["agents_future"]
    agents_id = pkl_data.get("agents_id")
    ego_index = pkl_data.get("ego_index")
    agent_types = pkl_data.get("agent_types")
    polylines = pkl_data.get("polylines", [])
    traffic_lights = pkl_data.get("traffic_light_points", np.zeros((0, 3), dtype=np.float32))

    def make_future_tensor(pred_traj: np.ndarray) -> np.ndarray:
        future = -np.ones_like(agents_future)
        if pred_traj.shape[0] == 0:
            return future
        # Use all available predicted agents for visualization so ego doesn't disappear
        n_pred_agents = min(pred_traj.shape[0], agents_future.shape[0])
        n_pred_timesteps = min(pred_traj.shape[1], agents_future.shape[1])
        valid_pred = pred_traj[:n_pred_agents, :n_pred_timesteps, :3]
        coord_mask = (
            (np.abs(valid_pred[:, :, 0]) < COORD_LIMIT)
            & (np.abs(valid_pred[:, :, 1]) < COORD_LIMIT)
        )
        future[:n_pred_agents, :n_pred_timesteps, :3] = valid_pred
        future[:n_pred_agents, :n_pred_timesteps, 0][~coord_mask] = -1
        future[:n_pred_agents, :n_pred_timesteps, 1][~coord_mask] = -1

        if n_pred_timesteps > 1:
            dt = 0.1
            pos_diff = np.diff(valid_pred[:, :, :2], axis=1)
            vel = pos_diff / dt
            vel_mask = np.abs(vel) < 100
            filtered_vel = np.where(vel_mask, vel, 0)
            future[:n_pred_agents, 1 : min(n_pred_timesteps, vel.shape[1] + 1), 3:5] = (
                filtered_vel[:, : min(n_pred_timesteps - 1, vel.shape[1])]
            )
        return future

    pred_futures = [(title, make_future_tensor(traj)) for title, traj in pred_trajs]

    images: List[np.ndarray] = []
    total_timesteps = agents_future.shape[1]

    if ego_index is None or not (0 <= int(ego_index) < agents_future.shape[0]):
        ego_index = 0
    ego_index = int(ego_index)

    ego_pos = agents_future[ego_index, :, :2]
    if pred_trajs:
        pred_traj = pred_trajs[0][1]
        if ego_index < pred_traj.shape[0]:
            ego_pos = pred_traj[ego_index, :, :2]

    ego_valid = (
        (ego_pos[:, 0] != 0)
        & (ego_pos[:, 0] != -1)
        & (np.abs(ego_pos[:, 0]) < COORD_LIMIT)
        & (np.abs(ego_pos[:, 1]) < COORD_LIMIT)
    )

    center_x = 0.0
    center_y = 0.0
    if np.any(ego_valid):
        first_valid = np.where(ego_valid)[0][0]
        center_x, center_y = ego_pos[first_valid]

    def agent_color(i: int, is_controlled: bool) -> str:
        if i == ego_index:
            return palette["ego_box"]
        if is_controlled:
            return palette["controlled_box"]
        if agent_types is None:
            return palette["vehicle_box"]
        agent_type = int(agent_types[i]) if i < len(agent_types) else 0
        if agent_type == 0:
            return palette["vehicle_box"]
        if agent_type == 1:
            return palette["ped_box"]
        if agent_type == 2:
            return palette["cyc_box"]
        return palette["other_box"]

    def plot_frame(
        ax: plt.Axes,
        future_full: np.ndarray,
        future_masked: np.ndarray,
        title: str,
        future_color: str,
        timestep: int,
        conditioning: np.ndarray | None,
    ) -> None:
        ax.set_facecolor("#FFFFFF")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(center_x - VIEW_RANGE, center_x + VIEW_RANGE)
        ax.set_ylim(center_y - VIEW_RANGE, center_y + VIEW_RANGE)
        ax.set_title(title, fontsize=12, pad=12, color="#333333")
        ax.set_xlabel("X (m)", color="#555555")
        ax.set_ylabel("Y (m)", color="#555555")
        ax.tick_params(colors="#666666", labelsize=8)
        ax.grid(color="#E0E0E0", linestyle="--", linewidth=0.5, alpha=0.6)

        for poly in polylines:
            arr = np.asarray(poly)
            if arr.size == 0:
                continue
            valid = (
                (arr[:, 0] != 0)
                & (np.abs(arr[:, 0]) < COORD_LIMIT)
                & (np.abs(arr[:, 1]) < COORD_LIMIT)
            )
            if np.any(valid):
                ax.plot(
                    arr[valid][:, 0],
                    arr[valid][:, 1],
                    color=palette["map"],
                    linewidth=0.8,
                    alpha=0.7,
                )

        num_agents = min(agents_history.shape[0], future_full.shape[0])

        for i in range(num_agents):
            if timestep < future_full.shape[1]:
                curr = future_full[i, timestep]
                if (
                    curr[0] != 0
                    and curr[0] != -1
                    and np.abs(curr[0]) < COORD_LIMIT
                    and np.abs(curr[1]) < COORD_LIMIT
                ):
                    heading = curr[2] if len(curr) > 2 and curr[2] != 0 else agents_history[i, -1, 2]

                    is_controlled = controlled_agent_ids is not None and i in controlled_agent_ids
                    box_color = agent_color(i, is_controlled)
                    box_alpha = 0.9 if i == ego_index else (0.8 if is_controlled else 0.65)

                    rect = mpatches.FancyBboxPatch(
                        (curr[0] - agents_history[i, -1, 5] / 2, curr[1] - agents_history[i, -1, 6] / 2),
                        agents_history[i, -1, 5],
                        agents_history[i, -1, 6],
                        ec="black",
                        fc=box_color,
                        linewidth=1.5,
                        alpha=box_alpha,
                        boxstyle=mpatches.BoxStyle("Round", pad=0.3),
                        zorder=5,
                    )
                    rect.set_transform(
                        transforms.Affine2D().rotate_around(curr[0], curr[1], heading) + ax.transData
                    )
                    ax.add_patch(rect)
                    heading_length = agents_history[i, -1, 5] / 2 + 1.5
                    end_x = curr[0] + heading_length * math.cos(heading)
                    end_y = curr[1] + heading_length * math.sin(heading)
                    ax.plot([curr[0], end_x], [curr[1], end_y], color="black", alpha=0.5, linewidth=1.0, zorder=6)
                    if show_agent_ids and agents_id is not None and i < len(agents_id):
                        label_offset = heading_length + 0.8
                        label_x = curr[0] + label_offset * math.cos(heading)
                        label_y = curr[1] + label_offset * math.sin(heading)
                        ax.text(
                            label_x,
                            label_y,
                            str(int(agents_id[i])),
                            color="#111111",
                            fontsize=5,
                            ha="center",
                            va="center",
                            zorder=6,
                            bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="#111111", lw=0.4, alpha=0.7),
                        )

            future = future_masked[i, timestep:]
            valid_future = (
                (future[:, 0] != 0)
                & (future[:, 0] != -1)
                & (np.abs(future[:, 0]) < COORD_LIMIT)
                & (np.abs(future[:, 1]) < COORD_LIMIT)
            )
            if np.sum(valid_future) > 1:
                seg = future[valid_future]
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    color=future_color,
                    linewidth=2.2,
                    alpha=0.95,
                    linestyle="--",
                )

            if conditioning is not None and i < conditioning.shape[0] and timestep < conditioning.shape[1]:
                cond_future = conditioning[i, timestep:]
                valid_cond = (
                    (cond_future[:, 0] != 0)
                    & (cond_future[:, 0] != -1)
                    & (np.abs(cond_future[:, 0]) < COORD_LIMIT)
                    & (np.abs(cond_future[:, 1]) < COORD_LIMIT)
                )
                if np.sum(valid_cond) > 1:
                    cond_seg = cond_future[valid_cond]
                    ax.plot(
                        cond_seg[:, 0],
                        cond_seg[:, 1],
                        color=palette["condition_future"],
                        linewidth=2.0,
                        alpha=0.7,
                        linestyle="-",
                    )

        if traffic_lights is not None and traffic_lights.size > 0:
            colors = []
            for tl in traffic_lights:
                state = int(tl[2]) if tl.shape[0] >= 3 else 0
                if state in [1, 4, 7]:
                    colors.append("#FF5252")
                elif state in [2, 5, 8]:
                    colors.append("#FFC107")
                elif state in [3, 6]:
                    colors.append("#4CAF50")
                else:
                    colors.append("#9E9E9E")
            ax.scatter(
                traffic_lights[:, 0],
                traffic_lights[:, 1],
                c=colors,
                s=20,
                alpha=0.85,
                edgecolors="none",
                zorder=4,
            )

    panel_titles: List[Tuple[str, np.ndarray, str]] = [
        (f"{scenario_name} - GT", agents_future, palette["gt_future"])
    ]
    alt_colors = [
        palette["pred_future"],
        palette["pred_future_alt"],
    ]
    for idx, (title, future) in enumerate(pred_futures):
        panel_titles.append((title, future, alt_colors[idx % len(alt_colors)]))

    layout_params = None
    fixed_dpi = 100

    for t in range(0, total_timesteps, 2):
        if np.any(ego_valid):
            idx = min(t, len(ego_pos) - 1)
            while idx >= 0 and not ego_valid[idx]:
                idx -= 1
            if idx >= 0:
                center_x, center_y = ego_pos[idx]

        fig, axes = plt.subplots(1, len(panel_titles), figsize=(6 * len(panel_titles), 6), dpi=fixed_dpi)
        axes = np.atleast_1d(axes)

        for ax, (title, future_full, color) in zip(axes, panel_titles):
            partial = future_full.copy()
            if t + 1 < partial.shape[1]:
                partial[:, t + 1 :] = -1
            overlay = conditioning_future if title.endswith("Control") else None
            plot_frame(
                ax,
                future_full,
                partial,
                f"{title} t={t * 0.1:.1f}s",
                color,
                t,
                overlay,
            )

        if layout_params is None:
            fig.tight_layout(pad=1.2)
            layout_params = {
                "left": fig.subplotpars.left,
                "right": fig.subplotpars.right,
                "top": fig.subplotpars.top,
                "bottom": fig.subplotpars.bottom,
                "wspace": fig.subplotpars.wspace,
                "hspace": fig.subplotpars.hspace,
            }
        else:
            fig.subplots_adjust(**layout_params)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img[:, :, :3].copy())
        plt.close(fig)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gif_path = f"{save_path}.gif"
    pil_images = [Image.fromarray(img) for img in images]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    print(f"Saved GIF to {gif_path}")

    plt.rcParams["axes.prop_cycle"] = original_style
    plt.style.use("default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CATK closed-loop visualization")
    parser.add_argument(
        "--base_ckpt_path", type=str, default="./ckpt/clsft_E9.ckpt"
    )
    parser.add_argument("--ctrl_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--only_controlnet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip loading/running base model and only run controlnet.",
    )
    parser.add_argument(
        "--catk_data_dir", type=str, default="./cache/SMART/validation"
    )
    parser.add_argument(
        "--tfrecord_dir",
        type=str,
        default="./cache/SMART/validation_tfrecords_splitted",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scenario_id", type=str, nargs="*", default=None)
    parser.add_argument(
        "--scenario_id_dir",
        type=str,
        default=None,
        help="Directory with files named like scenario_<id>_*.pkl to match scenarios.",
    )
    parser.add_argument("--max_scenarios", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cfg_guidance_mode",
        choices=["off", "auto", "manual"],
        default="off",
        help=(
            "Classifier-free guidance for autoregressive token logits. "
            "auto uses paper defaults by modality; manual requires --cfg_guidance_scale."
        ),
    )
    parser.add_argument(
        "--cfg_guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale omega for logit mixing.",
    )
    parser.add_argument(
        "--conditioning_mode",
        choices=["sketch", "latent", "text"],
        default=None,
        help="Conditioning mode for the control model (default: use checkpoint config).",
    )
    parser.add_argument(
        "--latent",
        choices=["none", "gt", "custom"],
        default="none",
        help="Shortcut for latent conditioning (gt/custom).",
    )
    parser.add_argument(
        "--latent_dir",
        type=str,
        default=None,
        help="Directory with scenario_<id>_latents.pkl for latent conditioning.",
    )
    parser.add_argument(
        "--latent_target_agents",
        type=str,
        default="",
        help="Comma-separated agent indices to receive latent conditioning.",
    )
    parser.add_argument("--sketch", choices=["gt", "right_turn", "custom"], default="gt")
    parser.add_argument("--forward_distance", type=float, default=5.0)
    parser.add_argument("--turn_distance", type=float, default=5.0)
    parser.add_argument("--turn_angle_deg", type=float, default=60.0)
    parser.add_argument("--target_agents", type=str, default="0,1")
    parser.add_argument(
        "--custom_sketch_file",
        type=str,
        default=None,
        help="Optional JSON/PKL file with custom relative sketch points.",
    )
    parser.add_argument(
        "--custom_sketch_view_range",
        type=float,
        default=60.0,
        help="Meters shown in interactive custom sketch local view.",
    )
    parser.add_argument(
        "--custom_sketch_min_points",
        type=int,
        default=3,
        help="Minimum clicked points required for one custom sketch.",
    )
    parser.add_argument(
        "--custom_sketch_save_inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-scenario drawn relative sketch points under output_dir/custom_sketch_inputs.",
    )
    parser.add_argument(
        "--text",
        choices=["none", "gt", "custom"],
        default="none",
        help="Text conditioning mode (gt uses ProSIM instructions).",
    )
    parser.add_argument(
        "--custom_text",
        type=str,
        default="",
        help="Custom text prompt with <ego>/<agent_id> markers. Use --custom_text_delim to split.",
    )
    parser.add_argument(
        "--custom_text_file",
        type=str,
        default=None,
        help="Path to a text file (one sentence per line) for custom text prompts.",
    )
    parser.add_argument(
        "--custom_text_delim",
        type=str,
        default="||",
        help="Delimiter used to split --custom_text into sentences.",
    )
    parser.add_argument(
        "--debug_print_prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-agent final text prompts used for conditioning.",
    )
    parser.add_argument(
        "--prosim_instruct_dir",
        type=str,
        default=None,
        help="Path to cache root containing SMART/prosim_instruct_520k and mapping json.",
    )
    parser.add_argument(
        "--text_prompt_subdir",
        type=str,
        default="70b_0.8_0.9_action_prompt_v2",
        help="Subdir under text_prompts.",
    )
    parser.add_argument(
        "--text_mode",
        type=str,
        default="tag_only",
        choices=["full", "tag_only", "text_only"],
        help="Which GT text sources to include: full, tag_only, or text_only.",
    )
    parser.add_argument(
        "--text_split",
        type=str,
        default="val",
        help="text_prompts split to use (train/val).",
    )
    parser.add_argument(
        "--tag_prompt_subdir",
        type=str,
        default="auto",
        help="tag_prompts subdir to use (or auto to match the split).",
    )
    parser.add_argument(
        "--tag_sentence_style",
        type=str,
        default="ordered",
        choices=["ordered", "detailed"],
        help="How to render tag prompts into text.",
    )
    parser.add_argument(
        "--tag_time_scale",
        type=float,
        default=0.1,
        help="Scale for tag time indices to seconds.",
    )
    parser.add_argument(
        "--control_mask_mode",
        choices=["train_mask", "sketch", "target_agent_only"],
        default="train_mask",
        help=(
            "How to build control_mask for visualization. "
            "target_agent_only uses Waymo tracks_to_predict agents (agent.role[:, 2])."
        ),
    )
    parser.add_argument(
        "--control_max_num",
        type=int,
        default=32,
        help="Max number of controlled agents (train_mask mode).",
    )
    parser.add_argument(
        "--save_wosac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save WOSAC-compatible PKL rollouts.",
    )
    parser.add_argument(
        "--save_vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save visualization GIFs.",
    )
    parser.add_argument(
        "--wosac_num_rollouts",
        type=int,
        default=32,
        help="Number of rollouts to save when --save_wosac is enabled.",
    )
    return parser.parse_args()


def _parse_target_agents(target_agents: str) -> Tuple[int, ...]:
    if not target_agents:
        return ()
    out = []
    for part in target_agents.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return tuple(out)


_CFG_GUIDANCE_DEFAULTS = {
    "sketch": 0.5,
    "latent": 0.7,
    "text": 1.0,
}


def _infer_cfg_modality(args: argparse.Namespace) -> str:
    if args.text != "none":
        return "text"
    if args.latent != "none" or args.conditioning_mode == "latent":
        return "latent"
    return "sketch"


def _resolve_cfg_guidance(args: argparse.Namespace) -> Tuple[bool, float, str]:
    mode = str(args.cfg_guidance_mode).lower()
    modality = _infer_cfg_modality(args)
    if mode == "off":
        return False, 1.0, modality
    if args.cfg_guidance_scale is not None:
        return True, float(args.cfg_guidance_scale), modality
    if mode == "auto":
        return True, float(_CFG_GUIDANCE_DEFAULTS[modality]), modality
    raise RuntimeError("--cfg_guidance_scale is required when --cfg_guidance_mode=manual")


def run_visualization(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    base_model = None
    if not args.only_controlnet:
        base_model = load_model_from_checkpoint(
            args.base_ckpt_path, SMARTBase, device, output_dir
        )
    ctrl_model = None
    if args.ctrl_ckpt_path:
        ctrl_model = load_model_from_checkpoint(
            args.ctrl_ckpt_path, SMARTCtrl, device, output_dir
        )
    if ctrl_model is None and base_model is None:
        raise RuntimeError("Either --base_ckpt_path or --ctrl_ckpt_path must be provided.")

    # Prefer ctrl_model's config when available (it has control-specific fields);
    # otherwise fall back to base_model's config for base-only inference.
    cfg_model = ctrl_model if ctrl_model is not None else base_model
    model_cfg = cfg_model.hparams.model_config
    num_historical_steps = int(model_cfg.decoder.num_historical_steps)
    num_future_steps = int(model_cfg.decoder.num_future_steps)
    control_future_len = int(model_cfg.get("control", {}).get("future_len", 16))
    ctrl_mode_raw = model_cfg.get("control", {}).get("conditioning_mode", None)
    if ctrl_mode_raw is None or str(ctrl_mode_raw).lower() in {"", "none", "null"}:
        ctrl_mode = "sketch"
    else:
        ctrl_mode = str(ctrl_mode_raw).lower()
        if ctrl_mode == "future":
            ctrl_mode = "sketch"
    if args.latent != "none":
        if args.conditioning_mode is not None and args.conditioning_mode != "latent":
            raise RuntimeError(
                "Use either --latent or --conditioning_mode, not both."
            )
        args.conditioning_mode = "latent"
    if args.conditioning_mode is None:
        args.conditioning_mode = ctrl_mode
    elif args.conditioning_mode != ctrl_mode:
        raise RuntimeError(
            f"conditioning_mode mismatch: ckpt={ctrl_mode}, args={args.conditioning_mode}"
        )
    latent_dir = Path(args.latent_dir) if args.latent_dir else None
    if args.conditioning_mode == "latent" and latent_dir is None:
        raise RuntimeError("--latent_dir is required when conditioning_mode=latent")
    cfg_guidance_enabled, cfg_guidance_scale, cfg_guidance_modality = (
        _resolve_cfg_guidance(args)
    )
    if cfg_guidance_enabled:
        print(
            "[CFG] classifier-free guidance enabled: "
            f"modality={cfg_guidance_modality}, scale={cfg_guidance_scale}"
        )
        if ctrl_model is None:
            print("[CFG] ignored because no control checkpoint is loaded.")

    catk_dir = Path(args.catk_data_dir)
    scenario_ids = args.scenario_id
    if args.scenario_id_dir:
        scenario_ids = list_scenario_ids_from_dir(Path(args.scenario_id_dir))
        if not scenario_ids:
            raise RuntimeError(
                f"No scenario ids found in directory: {args.scenario_id_dir}"
            )
    text_mapping = None
    text_prompt_root = None
    custom_sentences = None
    if args.text == "gt":
        if args.prosim_instruct_dir is None:
            raise RuntimeError("--prosim_instruct_dir is required when --text=gt")
        cache_root = Path(args.prosim_instruct_dir)
        mapping_path = cache_root / "SMART" / "validation_scenario_to_prosim_scene_strict.json"
        text_mapping = _load_text_mapping(mapping_path)
        text_prompt_root = cache_root / "SMART" / "prosim_instruct_520k"
        if scenario_ids is None:
            scenario_ids = list(text_mapping.keys())
        if scenario_ids is not None:
            before_count = len(scenario_ids)
            scenario_ids = [sid for sid in scenario_ids if sid in text_mapping]
            after_count = len(scenario_ids)
            print(f"[TEXT-GT] scenarios filtered by mapping: {before_count} -> {after_count}")
    elif args.text == "custom":
        custom_text_file = Path(args.custom_text_file) if args.custom_text_file else None
        custom_sentences = _parse_custom_text(
            args.custom_text, custom_text_file, args.custom_text_delim
        )
        if not custom_sentences:
            raise RuntimeError(
                "Custom text is empty. Use --custom_text or --custom_text_file."
            )
    data_paths = select_catk_scenarios(
        catk_dir, args.seed, args.max_scenarios, scenario_ids
    )
    if not data_paths:
        raise RuntimeError("No CATK scenarios found for visualization")

    dataset = ScenarioDataset(data_paths)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    tfrecord_dir = Path(args.tfrecord_dir)
    target_agents = _parse_target_agents(args.target_agents)
    custom_sketch_file = Path(args.custom_sketch_file) if args.custom_sketch_file else None
    if args.sketch == "custom" and args.conditioning_mode == "sketch" and not target_agents:
        raise RuntimeError("--target_agents is required when --sketch=custom")

    total_scenarios = len(data_paths)
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=total_scenarios, desc="Scenarios", unit="sc")

    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device)
            tok_source = ctrl_model if ctrl_model is not None else base_model
            tokenized_map, tokenized_agent = tok_source.token_processor(batch)

            scenario_id = batch["scenario_id"][0]
            if isinstance(scenario_id, bytes):
                scenario_id = scenario_id.decode("utf-8")

            def _build_gt_sketch_aligned() -> torch.Tensor:
                """Match training-time sketch construction when possible."""
                if ctrl_model is None:
                    return build_gt_sketch(tokenized_agent, control_future_len)
                builder = getattr(ctrl_model.encoder.agent_encoder, "_build_control_sketch", None)
                if callable(builder):
                    return builder(tokenized_agent)
                return build_gt_sketch(tokenized_agent, control_future_len)

            control_text = None
            control_text_mask = None
            if args.text == "gt":
                if scenario_id not in text_mapping:
                    continue
                control_text, control_text_mask = _build_control_text_for_batch(
                    batch=batch,
                    scenario_id=scenario_id,
                    text_prompt_root=text_prompt_root,
                    text_prompt_subdir=args.text_prompt_subdir,
                    text_split=str(args.text_split).lower(),
                    tag_prompt_subdir=str(args.tag_prompt_subdir).lower(),
                    tag_time_scale=args.tag_time_scale,
                    text_mode=str(args.text_mode).lower(),
                    tag_sentence_style=str(args.tag_sentence_style).lower(),
                    text_mapping=text_mapping,
                )
            elif args.text == "custom":
                control_text, control_text_mask = _build_control_text_from_custom(
                    batch=batch,
                    sentences=custom_sentences or [],
                )
                if control_text_mask.sum().item() == 0:
                    print(
                        f"[TEXT-CUSTOM] no agents matched in scenario {scenario_id}"
                    )
            if args.debug_print_prompts and control_text is not None:
                agent_ids = batch["agent"]["id"].detach().cpu().tolist()
                for idx, agent_id in enumerate(agent_ids):
                    if idx >= len(control_text):
                        prompt = "[MISSING]"
                    else:
                        prompt = control_text[idx] or "[EMPTY]"
                    print(f"[TEXT-PROMPT][{scenario_id}] agent {agent_id}: {prompt}")

            control_sketch = None
            control_latents = None
            control_scene_latents = None
            controlled_ids: List[int] = []
            control_mask = None
            use_controlled_ids_for_mask = False
            custom_points_by_agent: Dict[int, np.ndarray] = {}

            def _build_custom_sketch_for_scenario():
                polylines_world: List[np.ndarray] = []
                try:
                    map_save = batch["map_save"]
                except Exception:
                    map_save = None
                if map_save is not None:
                    polylines_world = build_polylines_from_map_save(map_save)
                return build_custom_sketch(
                    batch=batch,
                    scenario_id=str(scenario_id),
                    num_historical_steps=num_historical_steps,
                    future_len=control_future_len,
                    target_agents=target_agents,
                    polylines_world=polylines_world,
                    custom_sketch_file=custom_sketch_file,
                    custom_view_range=args.custom_sketch_view_range,
                    custom_min_points=args.custom_sketch_min_points,
                )

            if args.conditioning_mode == "sketch":
                if args.sketch == "gt":
                    control_sketch = _build_gt_sketch_aligned()
                    controlled_ids = extract_controlled_agent_indices(
                        control_sketch.detach().cpu().numpy()
                    )
                elif args.sketch == "right_turn":
                    control_sketch, controlled_ids = build_right_turn_sketch(
                        batch=batch,
                        num_historical_steps=num_historical_steps,
                        future_len=control_future_len,
                        forward_distance=args.forward_distance,
                        turn_distance=args.turn_distance,
                        turn_angle_deg=args.turn_angle_deg,
                        target_agents=target_agents,
                    )
                else:
                    (
                        control_sketch,
                        controlled_ids,
                        custom_points_by_agent,
                    ) = _build_custom_sketch_for_scenario()
                    if not controlled_ids:
                        print(
                            f"[CUSTOM-SKETCH] scenario={scenario_id}: "
                            "no valid sketch drawn; skipping scenario."
                        )
                        continue
                use_controlled_ids_for_mask = True
            elif args.conditioning_mode == "latent":
                payload = load_latent_payload(latent_dir, scenario_id)
                control_latents, latent_mask, control_scene_latents = align_latents_to_agents(
                    payload, batch["agent"]["id"], device
                )
                control_mask = latent_mask
                if args.sketch == "gt":
                    control_sketch = _build_gt_sketch_aligned()
                elif args.sketch == "right_turn":
                    control_sketch, _ = build_right_turn_sketch(
                        batch=batch,
                        num_historical_steps=num_historical_steps,
                        future_len=control_future_len,
                        forward_distance=args.forward_distance,
                        turn_distance=args.turn_distance,
                        turn_angle_deg=args.turn_angle_deg,
                        target_agents=target_agents,
                    )
                elif args.sketch == "custom":
                    (
                        control_sketch,
                        _,
                        custom_points_by_agent,
                    ) = _build_custom_sketch_for_scenario()
                latent_targets = _parse_target_agents(args.latent_target_agents)
                if latent_targets:
                    n_agent = tokenized_agent["valid_mask"].shape[0]
                    target_mask = torch.zeros(n_agent, device=device, dtype=torch.bool)
                    for idx in latent_targets:
                        if 0 <= idx < n_agent:
                            target_mask[idx] = True
                    control_mask = target_mask if control_mask is None else (control_mask & target_mask)
                    use_controlled_ids_for_mask = True
            else:
                # text-only conditioning: handled via control_text/control_text_mask
                pass

            if controlled_ids and use_controlled_ids_for_mask:
                n_agent = tokenized_agent["valid_mask"].shape[0]
                mask = torch.zeros(n_agent, device=device, dtype=torch.bool)
                for idx in controlled_ids:
                    if 0 <= idx < n_agent:
                        mask[idx] = True
                control_mask = mask if control_mask is None else (control_mask & mask)

            if args.control_mask_mode == "target_agent_only":
                target_mask = build_waymo_target_control_mask(batch)
                if target_mask is None:
                    n_agent = tokenized_agent["valid_mask"].shape[0]
                    target_mask = torch.zeros(n_agent, device=device, dtype=torch.bool)
                    print(
                        f"[TARGET-ONLY] scenario={scenario_id}: "
                        "agent.role[:, 2] is unavailable; no agents controlled."
                    )
                target_mask = target_mask.to(device)
                control_mask = (
                    target_mask if control_mask is None else (control_mask & target_mask)
                )
                if control_sketch is not None:
                    control_sketch = control_sketch * target_mask[:, None, None].to(
                        control_sketch.dtype
                    )
                if control_text is not None:
                    target_mask_cpu = target_mask.detach().cpu().tolist()
                    control_text = [
                        prompt if idx < len(target_mask_cpu) and target_mask_cpu[idx] else ""
                        for idx, prompt in enumerate(control_text)
                    ]
                if control_text_mask is not None:
                    control_text_mask = control_text_mask & target_mask.detach().cpu()

            if args.control_mask_mode == "train_mask":
                train_mask = build_train_control_mask(
                    batch=batch,
                    max_num=args.control_max_num,
                    seed=args.seed,
                    scenario_id=scenario_id,
                )
                if train_mask is None:
                    n_agent = tokenized_agent["valid_mask"].shape[0]
                    train_mask = torch.zeros(
                        n_agent, device=device, dtype=torch.bool
                    )
                train_mask = train_mask.to(device)
                control_mask = train_mask if control_mask is None else (control_mask & train_mask)

            if control_text_mask is not None:
                control_text_mask = control_text_mask.to(device)
                control_mask = (
                    control_text_mask
                    if control_mask is None
                    else (control_mask & control_text_mask)
                )

            if control_mask is not None:
                controlled_ids = torch.where(control_mask)[0].tolist()

            if (
                args.sketch == "custom"
                and args.custom_sketch_save_inputs
                and custom_points_by_agent
            ):
                custom_input_dir = output_dir / "custom_sketch_inputs"
                custom_input_path = custom_input_dir / f"{scenario_id}.json"
                save_custom_sketch_points(
                    output_path=custom_input_path,
                    scenario_id=str(scenario_id),
                    points_by_agent=custom_points_by_agent,
                )

            base_pred = None
            if base_model is not None:
                base_pred = base_model.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    base_model.validation_rollout_sampling,
                )
                base_pred = apply_rollout_boundary_if_available(
                    base_model, batch, base_pred
                )
            ctrl_pred = None
            if ctrl_model is not None:
                ctrl_pred = ctrl_model.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    ctrl_model.validation_rollout_sampling,
                    control_sketch=control_sketch,
                    control_mask=control_mask,
                    control_latents=control_latents,
                    control_scene_latents=control_scene_latents,
                    control_text=control_text,
                    control_text_mask=control_text_mask,
                    cfg_guidance_enabled=cfg_guidance_enabled,
                    cfg_guidance_scale=cfg_guidance_scale,
                )
                ctrl_pred = apply_rollout_boundary_if_available(
                    ctrl_model, batch, ctrl_pred
                )

            base_pred_traj = None
            base_pred_head = None
            base_pred_z = None
            base_pred_valid = None
            if base_pred is not None:
                base_pred_traj = base_pred["pred_traj_10hz"].detach().cpu().numpy()
                base_pred_head = base_pred["pred_head_10hz"].detach().cpu().numpy()
                base_pred_z = base_pred.get("pred_z_10hz")
                base_pred_valid = base_pred.get("pred_valid_10hz")
            ctrl_pred_traj = None
            ctrl_pred_head = None
            ctrl_pred_z = None
            ctrl_pred_valid = None
            if ctrl_pred is not None:
                ctrl_pred_traj = ctrl_pred["pred_traj_10hz"].detach().cpu().numpy()
                ctrl_pred_head = ctrl_pred["pred_head_10hz"].detach().cpu().numpy()
                ctrl_pred_z = ctrl_pred.get("pred_z_10hz")
                ctrl_pred_valid = ctrl_pred.get("pred_valid_10hz")
            if base_pred_z is not None:
                base_pred_z = base_pred_z.detach().cpu().numpy()
            if base_pred_valid is not None:
                base_pred_valid = base_pred_valid.detach().cpu().numpy()
            if ctrl_pred_z is not None:
                ctrl_pred_z = ctrl_pred_z.detach().cpu().numpy()
            if ctrl_pred_valid is not None:
                ctrl_pred_valid = ctrl_pred_valid.detach().cpu().numpy()

            base_traj_xyyaw = None
            if base_pred_traj is not None and base_pred_head is not None:
                base_traj_xyyaw = make_pred_traj_xyyaw(base_pred_traj, base_pred_head)
            ctrl_traj_xyyaw = None
            if ctrl_pred_traj is not None and ctrl_pred_head is not None:
                ctrl_traj_xyyaw = make_pred_traj_xyyaw(ctrl_pred_traj, ctrl_pred_head)

            agent_batch = batch["agent"]["batch"]
            agent_mask = agent_batch == 0

            agents_history = None
            agents_future = None
            if args.save_vis:
                agents_history, agents_future = build_agents_history_future(
                    position=batch["agent"]["position"][agent_mask],
                    heading=batch["agent"]["heading"][agent_mask],
                    velocity=batch["agent"]["velocity"][agent_mask],
                    shape=batch["agent"]["shape"][agent_mask],
                    valid_mask=batch["agent"]["valid_mask"][agent_mask],
                    num_historical_steps=num_historical_steps,
                    num_future_steps=num_future_steps,
                )

                polylines = []
                traffic_lights = np.zeros((0, 3), dtype=np.float32)
                tfrecord_path = tfrecord_dir / f"{scenario_id}.tfrecords"
                if tfrecord_path.exists():
                    scenario = load_waymo_scenario(tfrecord_path)
                    if scenario is not None:
                        polylines = build_polylines(scenario)
                        traffic_lights = build_traffic_light_points(
                            scenario, num_historical_steps - 1
                        )
                if not polylines:
                    try:
                        map_save = batch["map_save"]
                    except Exception:
                        map_save = None
                    if map_save is not None:
                        polylines = build_polylines_from_map_save(map_save)

                agents_id = batch["agent"]["id"][agent_mask].detach().cpu().numpy()
                agent_types = batch["agent"]["type"][agent_mask].detach().cpu().numpy()
                role = batch["agent"]["role"][agent_mask].detach().cpu().numpy()
                ego_candidates = np.where(role[:, 0])[0] if role.ndim == 2 else []
                ego_index = int(ego_candidates[0]) if len(ego_candidates) > 0 else None
                scenario_data = {
                    "agents_history": agents_history,
                    "agents_future": agents_future,
                    "polylines": polylines,
                    "traffic_light_points": traffic_lights,
                    "agents_id": agents_id,
                    "agent_types": agent_types,
                    "ego_index": ego_index,
                }

                predictions_for_vis = []
                if base_traj_xyyaw is not None:
                    predictions_for_vis.append(("Base", base_traj_xyyaw))
                if ctrl_traj_xyyaw is not None:
                    predictions_for_vis.append(("Control", ctrl_traj_xyyaw))

                conditioning_future = None
                if control_sketch is not None:
                    conditioning_future = sketch_to_future(
                        control_sketch.detach().cpu().numpy(), agents_future.shape[1]
                    )

                tag = str(args.sketch)
                video_path = (output_dir / f"{scenario_id}_{tag}").as_posix()
                create_trajectory_video(
                    scenario_data,
                    predictions_for_vis,
                    video_path,
                    scenario_id,
                    conditioning_future,
                    controlled_ids,
                )

            if args.save_wosac:
                num_rollouts = max(1, int(args.wosac_num_rollouts))
                base_rollouts = [] if base_model is not None else None
                base_valids = [] if base_model is not None else None
                ctrl_rollouts = [] if ctrl_model is not None else None
                ctrl_valids = [] if ctrl_model is not None else None

                def _pred_to_np(pred_dict: Dict[str, torch.Tensor]):
                    traj = pred_dict["pred_traj_10hz"].detach().cpu().numpy()
                    head = pred_dict["pred_head_10hz"].detach().cpu().numpy()
                    pred_z_np = pred_dict.get("pred_z_10hz")
                    pred_valid_np = pred_dict.get("pred_valid_10hz")
                    if pred_z_np is not None:
                        pred_z_np = pred_z_np.detach().cpu().numpy()
                    if pred_valid_np is not None:
                        pred_valid_np = pred_valid_np.detach().cpu().numpy()
                    return traj, head, pred_z_np, pred_valid_np

                for rollout_idx in range(num_rollouts):
                    if rollout_idx == 0:
                        if ctrl_model is not None:
                            c_traj, c_head, c_z = (
                                ctrl_pred_traj,
                                ctrl_pred_head,
                                ctrl_pred_z,
                            )
                            c_valid = ctrl_pred_valid
                        if base_model is not None:
                            b_traj, b_head, b_z = (
                                base_pred_traj,
                                base_pred_head,
                                base_pred_z,
                            )
                            b_valid = base_pred_valid
                    else:
                        set_seed(args.seed + rollout_idx)
                        if base_model is not None:
                            base_pred_r = base_model.encoder.inference(
                                tokenized_map,
                                tokenized_agent,
                                base_model.validation_rollout_sampling,
                            )
                            base_pred_r = apply_rollout_boundary_if_available(
                                base_model, batch, base_pred_r
                            )
                            b_traj, b_head, b_z, b_valid = _pred_to_np(base_pred_r)
                        if ctrl_model is not None:
                            ctrl_pred_r = ctrl_model.encoder.inference(
                                tokenized_map,
                                tokenized_agent,
                                ctrl_model.validation_rollout_sampling,
                                control_sketch=control_sketch,
                                control_mask=control_mask,
                                control_latents=control_latents,
                                control_scene_latents=control_scene_latents,
                                control_text=control_text,
                                control_text_mask=control_text_mask,
                                cfg_guidance_enabled=cfg_guidance_enabled,
                                cfg_guidance_scale=cfg_guidance_scale,
                            )
                            ctrl_pred_r = apply_rollout_boundary_if_available(
                                ctrl_model, batch, ctrl_pred_r
                            )
                            c_traj, c_head, c_z, c_valid = _pred_to_np(ctrl_pred_r)

                    if ctrl_model is not None:
                        ctrl_rollout, ctrl_valid = build_wosac_rollout(
                            c_traj, c_head, c_z, c_valid
                        )
                        ctrl_rollouts.append(ctrl_rollout)
                        ctrl_valids.append(ctrl_valid)
                    if base_model is not None:
                        base_rollout, base_valid = build_wosac_rollout(
                            b_traj, b_head, b_z, b_valid
                        )
                        base_rollouts.append(base_rollout)
                        base_valids.append(base_valid)

                ctrl_rollout_arr = None
                ctrl_valid_arr = None
                if ctrl_model is not None:
                    ctrl_rollout_arr = np.stack(ctrl_rollouts, axis=0)
                    ctrl_valid_arr = np.stack(ctrl_valids, axis=0)
                base_rollout_arr = None
                base_valid_arr = None
                if base_model is not None:
                    base_rollout_arr = np.stack(base_rollouts, axis=0)
                    base_valid_arr = np.stack(base_valids, axis=0)

                agents_id = batch["agent"]["id"][agent_mask].detach().cpu().numpy()
                if agents_history is not None:
                    last_history_valid = np.any(
                        agents_history[:, -1, :2] != 0, axis=-1
                    )
                else:
                    agent_pos = batch["agent"]["position"][agent_mask]
                    agent_valid = batch["agent"]["valid_mask"][agent_mask]
                    hist_idx = min(num_historical_steps - 1, agent_pos.shape[1] - 1)
                    pos_hist = agent_pos[:, hist_idx, :2].detach().cpu().numpy()
                    valid_hist = agent_valid[:, hist_idx].detach().cpu().numpy().astype(bool)
                    last_history_valid = valid_hist & (np.any(pos_hist != 0, axis=-1))
                model_rollouts = {}
                if ctrl_rollout_arr is not None:
                    model_rollouts["controlnet"] = {
                        "rollouts": ctrl_rollout_arr,
                        "valid_mask": ctrl_valid_arr,
                    }
                if base_rollout_arr is not None and base_valid_arr is not None:
                    model_rollouts["baseline"] = {
                        "rollouts": base_rollout_arr,
                        "valid_mask": base_valid_arr,
                    }
                num_steps_ref = (
                    ctrl_rollout_arr.shape[2]
                    if ctrl_rollout_arr is not None
                    else base_rollout_arr.shape[2]
                )
                save_wosac_pkl(
                    output_dir=output_dir,
                    scenario_id=scenario_id,
                    rollouts=model_rollouts,
                    agents_id=agents_id,
                    sim_agent_mask=last_history_valid,
                    num_steps=num_steps_ref,
                    num_rollouts=num_rollouts,
                    control_sketch=control_sketch.detach().cpu().numpy()
                    if control_sketch is not None
                    else None,
                    controlled_agents=controlled_ids,
                    seed=args.seed,
                )


__all__ = ["parse_args", "run_visualization"]
