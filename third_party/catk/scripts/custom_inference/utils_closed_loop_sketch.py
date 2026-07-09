import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def extract_controlled_agent_indices(agents_future: np.ndarray | None) -> List[int]:
    if agents_future is None or agents_future.ndim < 3:
        return []
    spatial = np.abs(agents_future[..., :2])
    mask = np.any(spatial > 0.0, axis=-1)
    controlled: List[int] = []
    if mask.ndim == 2:
        for agent_idx in range(mask.shape[0]):
            if mask[agent_idx].any():
                controlled.append(agent_idx)
    elif mask.ndim == 3:
        for agent_idx in range(mask.shape[1]):
            if mask[:, agent_idx].any():
                controlled.append(agent_idx)
    return controlled


def build_gt_sketch(tokenized_agent: dict, future_len: int) -> torch.Tensor:
    pos = tokenized_agent["gt_pos_raw"]
    head = tokenized_agent["gt_head_raw"]
    valid = tokenized_agent.get("gt_valid_raw")
    if valid is None:
        valid = torch.ones_like(head, dtype=torch.bool)
    sketch = torch.zeros((pos.shape[0], future_len, 3), device=pos.device, dtype=pos.dtype)
    max_len = min(future_len, pos.shape[1])
    sketch[:, :max_len, 0:2] = pos[:, :max_len]
    sketch[:, :max_len, 2] = head[:, :max_len]
    if max_len < future_len:
        sketch[:, max_len:, 0:2] = pos[:, max_len - 1 : max_len]
        sketch[:, max_len:, 2] = head[:, max_len - 1 : max_len]
    if valid is not None:
        valid = valid[:, :max_len]
        sketch[:, :max_len][~valid] = 0.0
    return sketch


def build_right_turn_sketch(
    *,
    batch,
    num_historical_steps: int,
    future_len: int,
    forward_distance: float,
    turn_distance: float,
    turn_angle_deg: float,
    target_agents: Tuple[int, ...],
) -> Tuple[torch.Tensor, List[int]]:
    agent_pos = batch["agent"]["position"]
    agent_head = batch["agent"]["heading"]
    valid_mask = batch["agent"]["valid_mask"]
    step_idx = min(num_historical_steps - 1, agent_pos.shape[1] - 1)
    device = agent_pos.device
    num_agents = agent_pos.shape[0]
    sketch = torch.zeros((num_agents, future_len, 3), device=device, dtype=agent_pos.dtype)

    controlled_ids: List[int] = []
    if not target_agents:
        return sketch, controlled_ids

    turn_angle = math.radians(turn_angle_deg)
    for agent_idx in target_agents:
        if agent_idx < 0 or agent_idx >= num_agents:
            continue
        if not bool(valid_mask[agent_idx, step_idx]):
            continue
        start_x = agent_pos[agent_idx, step_idx, 0]
        start_y = agent_pos[agent_idx, step_idx, 1]
        heading = agent_head[agent_idx, step_idx]

        forward_dx = forward_distance * torch.cos(heading)
        forward_dy = forward_distance * torch.sin(heading)
        turn_heading = heading + turn_angle
        turn_dx = turn_distance * torch.cos(turn_heading)
        turn_dy = turn_distance * torch.sin(turn_heading)

        traj_x = torch.linspace(start_x, start_x + forward_dx, future_len, device=device)
        traj_y = torch.linspace(start_y, start_y + forward_dy, future_len, device=device)
        traj_yaw = torch.full((future_len,), heading, device=device)

        mid = max(1, future_len // 2)
        traj_x[mid:] = torch.linspace(
            start_x + forward_dx, start_x + forward_dx + turn_dx, future_len - mid, device=device
        )
        traj_y[mid:] = torch.linspace(
            start_y + forward_dy, start_y + forward_dy + turn_dy, future_len - mid, device=device
        )
        traj_yaw[mid:] = turn_heading

        sketch[agent_idx, :, 0] = traj_x[:future_len]
        sketch[agent_idx, :, 1] = traj_y[:future_len]
        sketch[agent_idx, :, 2] = traj_yaw[:future_len]
        controlled_ids.append(agent_idx)

    return sketch, controlled_ids


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_local(world_xy: np.ndarray, origin_xy: np.ndarray, heading: float) -> np.ndarray:
    if world_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    dx = world_xy[:, 0] - origin_xy[0]
    dy = world_xy[:, 1] - origin_xy[1]
    c = math.cos(heading)
    s = math.sin(heading)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return np.stack([x_local, y_local], axis=1).astype(np.float32)


def _local_to_world(local_xy: np.ndarray, origin_xy: np.ndarray, heading: float) -> np.ndarray:
    if local_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    c = math.cos(heading)
    s = math.sin(heading)
    x_world = origin_xy[0] + c * local_xy[:, 0] - s * local_xy[:, 1]
    y_world = origin_xy[1] + s * local_xy[:, 0] + c * local_xy[:, 1]
    return np.stack([x_world, y_world], axis=1).astype(np.float32)


def _resample_path(points_xy: np.ndarray, num_steps: int) -> np.ndarray:
    if points_xy.shape[0] == 0 or num_steps <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points_xy.shape[0] == 1:
        return np.repeat(points_xy[:1], num_steps, axis=0).astype(np.float32)

    seg = np.diff(points_xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total < 1e-6:
        return np.repeat(points_xy[:1], num_steps, axis=0).astype(np.float32)

    target = np.linspace(0.0, total, num_steps)
    x = np.interp(target, cum, points_xy[:, 0])
    y = np.interp(target, cum, points_xy[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _yaw_from_path(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if points_xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)

    dx = np.gradient(points_xy[:, 0])
    dy = np.gradient(points_xy[:, 1])
    yaw = np.arctan2(dy, dx)
    yaw = np.nan_to_num(yaw, nan=0.0, posinf=0.0, neginf=0.0)
    return yaw.astype(np.float32)


def _parse_points_array(points: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    arr = arr[:, :2]
    if not np.isfinite(arr).all():
        return None
    return arr


def _load_custom_points_from_file(
    custom_sketch_file: Path,
    scenario_id: str,
) -> Dict[int, np.ndarray]:
    if not custom_sketch_file.exists():
        raise RuntimeError(f"Custom sketch file not found: {custom_sketch_file}")

    suffix = custom_sketch_file.suffix.lower()
    if suffix in {".json", ".jsn"}:
        payload = json.loads(custom_sketch_file.read_text(encoding="utf-8"))
    elif suffix in {".pkl", ".pickle"}:
        import pickle

        with open(custom_sketch_file, "rb") as f:
            payload = pickle.load(f)
    else:
        raise RuntimeError(
            "Unsupported custom sketch file format. Use .json or .pkl/.pickle."
        )

    scenario_payload = None
    if isinstance(payload, dict):
        if "scenarios" in payload and isinstance(payload["scenarios"], dict):
            scenario_payload = payload["scenarios"].get(scenario_id)
        if scenario_payload is None:
            scenario_payload = payload.get(scenario_id)
        if scenario_payload is None and "agents" in payload:
            scenario_payload = payload
    if scenario_payload is None:
        return {}

    agents = scenario_payload.get("agents", scenario_payload)
    if not isinstance(agents, dict):
        return {}

    out: Dict[int, np.ndarray] = {}
    for key, value in agents.items():
        try:
            agent_idx = int(key)
        except Exception:
            continue
        parsed = _parse_points_array(np.asarray(value))
        if parsed is not None:
            out[agent_idx] = parsed
    return out


def _filter_local_polylines(
    polylines_world: List[np.ndarray],
    origin_xy: np.ndarray,
    heading: float,
    view_range: float,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    margin = max(5.0, 0.25 * view_range)
    bound = view_range + margin
    for line in polylines_world:
        arr = np.asarray(line, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        local = _world_to_local(arr[:, :2], origin_xy, heading)
        in_range = (
            (np.abs(local[:, 0]) <= bound) & (np.abs(local[:, 1]) <= bound)
        )
        if np.any(in_range):
            out.append(local)
    return out


def _draw_custom_path_interactive(
    *,
    scenario_id: str,
    agent_idx: int,
    agent_id: int,
    history_local: np.ndarray,
    map_local_polylines: List[np.ndarray],
    view_range: float,
    min_points: int,
) -> Optional[np.ndarray]:
    import matplotlib
    backend = str(matplotlib.get_backend()).lower()

    def _is_non_gui_backend(name: str) -> bool:
        # Only block truly non-GUI backends. TkAgg/QtAgg are interactive and valid.
        non_gui = {
            "agg",
            "pdf",
            "ps",
            "svg",
            "pgf",
            "cairo",
            "template",
            "module://matplotlib_inline.backend_inline",
            "module://ipykernel.pylab.backend_inline",
        }
        return name in non_gui

    if _is_non_gui_backend(backend):
        # Try switching to an interactive backend before failing.
        candidates = ["tkagg", "qtagg", "qt5agg"]
        if os.environ.get("DISPLAY", ""):
            for cand in candidates:
                try:
                    matplotlib.use(cand, force=True)
                    backend = str(matplotlib.get_backend()).lower()
                    if not _is_non_gui_backend(backend):
                        break
                except Exception:
                    continue
        if _is_non_gui_backend(backend):
            if not os.environ.get("DISPLAY", ""):
                raise RuntimeError(
                    "Interactive custom sketch requires a GUI display, but DISPLAY is empty. "
                    "Run from a desktop/forwarded X11 shell (e.g. ssh -X), or provide "
                    "--custom_sketch_file to run headless."
                )
            raise RuntimeError(
                "Interactive custom sketch requires a GUI matplotlib backend. "
                f"Current backend: {backend}. "
                "Please install/use TkAgg or QtAgg."
            )

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_title(
        f"scenario={scenario_id} | agent_idx={agent_idx} | agent_id={agent_id}\n"
        "Left click: add point, Right click: remove, Middle/Enter: finish"
    )

    for line in map_local_polylines:
        ax.plot(line[:, 0], line[:, 1], color="#B0B0B0", linewidth=0.9, alpha=0.8)

    if history_local.shape[0] >= 2:
        ax.plot(
            history_local[:, 0],
            history_local[:, 1],
            color="#4C78A8",
            linewidth=2.0,
            alpha=0.9,
            label="history",
        )
    ax.scatter([0.0], [0.0], s=50, color="#D62728", label="current pose")
    ax.arrow(
        0.0,
        0.0,
        min(8.0, 0.2 * view_range),
        0.0,
        head_width=max(0.7, 0.02 * view_range),
        head_length=max(1.0, 0.03 * view_range),
        color="#D62728",
        linewidth=2.0,
        length_includes_head=True,
    )

    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper right")

    print(
        "[CUSTOM-SKETCH] scenario="
        f"{scenario_id}, agent_idx={agent_idx}, agent_id={agent_id}: "
        "draw local trajectory, then press Enter."
    )
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    if len(points) < min_points:
        print(
            "[CUSTOM-SKETCH] skipped "
            f"scenario={scenario_id}, agent_idx={agent_idx}: "
            f"need >= {min_points} points, got {len(points)}"
        )
        return None

    arr = np.asarray(points, dtype=np.float32)
    if np.linalg.norm(arr[0]) > 1e-3:
        arr = np.concatenate([np.zeros((1, 2), dtype=np.float32), arr], axis=0)
    return arr


def build_custom_sketch(
    *,
    batch,
    scenario_id: str,
    num_historical_steps: int,
    future_len: int,
    target_agents: Tuple[int, ...],
    polylines_world: Optional[List[np.ndarray]] = None,
    custom_sketch_file: Optional[Path] = None,
    custom_view_range: float = 60.0,
    custom_min_points: int = 3,
) -> Tuple[torch.Tensor, List[int], Dict[int, np.ndarray]]:
    agent_pos = batch["agent"]["position"]
    agent_head = batch["agent"]["heading"]
    valid_mask = batch["agent"]["valid_mask"]
    agent_id_tensor = batch["agent"].get("id", None)

    step_idx = min(num_historical_steps - 1, agent_pos.shape[1] - 1)
    hist_len = min(num_historical_steps, agent_pos.shape[1])
    device = agent_pos.device
    num_agents = agent_pos.shape[0]

    sketch = torch.zeros((num_agents, future_len, 3), device=device, dtype=agent_pos.dtype)
    controlled_ids: List[int] = []
    used_points: Dict[int, np.ndarray] = {}

    if not target_agents:
        return sketch, controlled_ids, used_points

    points_by_agent: Dict[int, np.ndarray] = {}
    if custom_sketch_file is not None:
        points_by_agent = _load_custom_points_from_file(custom_sketch_file, scenario_id)

    polylines_world = polylines_world or []

    for agent_idx in target_agents:
        if agent_idx < 0 or agent_idx >= num_agents:
            continue
        if not bool(valid_mask[agent_idx, step_idx]):
            continue

        local_points = points_by_agent.get(agent_idx)
        if local_points is None:
            origin_xy = (
                agent_pos[agent_idx, step_idx, :2].detach().cpu().numpy().astype(np.float32)
            )
            heading = float(agent_head[agent_idx, step_idx].detach().cpu().item())

            hist_world = (
                agent_pos[agent_idx, :hist_len, :2].detach().cpu().numpy().astype(np.float32)
            )
            hist_valid = (
                valid_mask[agent_idx, :hist_len].detach().cpu().numpy().astype(bool)
            )
            hist_world = hist_world[hist_valid]
            history_local = _world_to_local(hist_world, origin_xy, heading)
            local_polylines = _filter_local_polylines(
                polylines_world,
                origin_xy,
                heading,
                custom_view_range,
            )

            agent_id = int(agent_id_tensor[agent_idx].detach().cpu().item()) if agent_id_tensor is not None else agent_idx
            local_points = _draw_custom_path_interactive(
                scenario_id=scenario_id,
                agent_idx=agent_idx,
                agent_id=agent_id,
                history_local=history_local,
                map_local_polylines=local_polylines,
                view_range=custom_view_range,
                min_points=custom_min_points,
            )
            if local_points is None:
                continue

        sampled_local = _resample_path(local_points, future_len)
        local_yaw = _yaw_from_path(sampled_local)

        origin_xy = (
            agent_pos[agent_idx, step_idx, :2].detach().cpu().numpy().astype(np.float32)
        )
        heading = float(agent_head[agent_idx, step_idx].detach().cpu().item())
        sampled_world = _local_to_world(sampled_local, origin_xy, heading)
        world_yaw = _wrap_to_pi(local_yaw + heading).astype(np.float32)

        sketch[agent_idx, :, 0] = torch.from_numpy(sampled_world[:, 0]).to(
            device=device, dtype=agent_pos.dtype
        )
        sketch[agent_idx, :, 1] = torch.from_numpy(sampled_world[:, 1]).to(
            device=device, dtype=agent_pos.dtype
        )
        sketch[agent_idx, :, 2] = torch.from_numpy(world_yaw).to(
            device=device, dtype=agent_pos.dtype
        )

        controlled_ids.append(agent_idx)
        used_points[agent_idx] = local_points.astype(np.float32)

    return sketch, controlled_ids, used_points


def save_custom_sketch_points(
    output_path: Path,
    scenario_id: str,
    points_by_agent: Dict[int, np.ndarray],
) -> None:
    payload = {
        "scenario_id": scenario_id,
        "agents": {
            str(agent_idx): points[:, :2].astype(np.float32).tolist()
            for agent_idx, points in points_by_agent.items()
            if points.ndim == 2 and points.shape[1] >= 2 and points.shape[0] >= 2
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


__all__ = [
    "extract_controlled_agent_indices",
    "build_gt_sketch",
    "build_right_turn_sketch",
    "build_custom_sketch",
    "save_custom_sketch_points",
]
