import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

_VBD_ROOT = Path(__file__).resolve().parents[1]
if str(_VBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_VBD_ROOT))

from vbd.model.behavior_vae import BehaviorVAE
from vbd.model.model_utils import update_batch_for_training
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar dependency
    def tqdm(iterable, **kwargs):
        return iterable

""" 
cd third_party/VBD
python script/create_vbd_ctrl_latents.py \
    --dataset_root ./data/waymovalid-scenarionet \
    --ckpt ../../checkpoints/vbd/behaviorvae.ckpt \
    --cfg config/BehaviorVAE_full.yaml
    
python script/create_vbd_ctrl_latents.py \
    --dataset_root ./data/waymovalid-scenarionet \
    --latent_dir_name latents_behaviorvae_wcontext \
    --ckpt ../../checkpoints/vbd/behaviorvae.ckpt \
    --cfg config/BehaviorVAE_full.yaml
"""



def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


@dataclass
class WindowSpec:
    exec_h: int
    pred_h: int

    @property
    def state_len(self) -> int:
        return self.pred_h + 1


def _iter_windows(total_future_len: int, spec: WindowSpec) -> list[int]:
    """Return window start indices for receding-horizon with stride exec_h and horizon pred_h."""
    starts: list[int] = []
    t = 0
    while t + spec.pred_h < total_future_len:
        starts.append(t)
        t += spec.exec_h
    return starts


def _compute_anchors(agents_type: np.ndarray, anchors_dict: Dict[str, Any]) -> np.ndarray:
    """Match WaymaxDataset._process behavior for anchor vectors."""
    out = []
    for t in agents_type.tolist():
        if int(t) == 1:
            out.append(anchors_dict["TYPE_VEHICLE"])
        elif int(t) == 2:
            out.append(anchors_dict["TYPE_PEDESTRIAN"])
        elif int(t) == 3:
            out.append(anchors_dict["TYPE_CYCLIST"])
        else:
            out.append(np.zeros_like(anchors_dict["TYPE_VEHICLE"]))
    return np.asarray(out, dtype=np.float32)


def _to_torch_batch(sample: Dict[str, Any], *, agents_future_window: np.ndarray) -> Dict[str, torch.Tensor]:
    """Build a torch batch [B=1,...] for BehaviorVAE from a VBD sample dict."""
    out: Dict[str, torch.Tensor] = {}
    for k in [
        "agents_history",
        "agents_future",
        "agents_interested",
        "agents_type",
        "anchors",
        "traffic_light_points",
        "polylines",
        "polylines_valid",
        "relations",
    ]:
        if k not in sample:
            continue
        if k == "agents_future":
            out[k] = torch.from_numpy(agents_future_window)
        else:
            out[k] = torch.from_numpy(sample[k])
    # Add batch dim
    out = {k: v.unsqueeze(0) for k, v in out.items()}
    return out


@torch.no_grad()
def _generate_latents_for_dir(
    *,
    vbd_dir: str,
    out_latent_dir: str,
    model: BehaviorVAE,
    anchor_dict: Dict[str, Any],
    device: torch.device,
    spec: WindowSpec,
    agents_len: int,
    sample_latent: bool,
    overwrite: bool,
):
    os.makedirs(out_latent_dir, exist_ok=True)
    files = [f for f in os.listdir(vbd_dir) if f.endswith(".pkl")]
    files.sort()
    if not files:
        raise RuntimeError(f"No .pkl files found in: {vbd_dir}")

    for fname in tqdm(files, desc=os.path.basename(vbd_dir), unit="file"):
        in_path = os.path.join(vbd_dir, fname)
        with open(in_path, "rb") as f:
            sample = pickle.load(f)

        scenario_id = sample.get("scenario_id")
        if not scenario_id:
            # Fallback to filename
            scenario_id = os.path.splitext(fname)[0].replace("scenario_", "")

        out_path = os.path.join(out_latent_dir, f"scenario_{scenario_id}_latents.pkl")
        if os.path.exists(out_path) and not overwrite:
            continue

        agents_future = sample["agents_future"]  # [N,T,5]
        total_future_len = int(agents_future.shape[1])
        starts = _iter_windows(total_future_len, spec)

        # Add anchors field expected by the VBD Encoder.
        if "anchors" not in sample:
            sample = dict(sample)
            sample["anchors"] = _compute_anchors(sample["agents_type"], anchor_dict)

        # Initialize working batch for closed-loop-style context updates.
        init_window = agents_future[:, : spec.state_len, :]
        working_batch = _to_torch_batch(sample, agents_future_window=init_window)
        working_batch = {k: v.to(device) for k, v in working_batch.items()}

        z_segments: list[np.ndarray] = []
        for s in tqdm(starts, desc=f"{scenario_id}", unit="win", leave=False):
            window = agents_future[:, s : s + spec.state_len, :]
            working_batch["agents_future"] = torch.from_numpy(window).unsqueeze(0).to(
                device=device, dtype=working_batch["agents_history"].dtype
            )

            out = model(working_batch, sample_latent=sample_latent)
            z = out.z_agent[0, :agents_len].detach().cpu().numpy()  # [A,Za]
            z_segments.append(z)

            # Update context for the next window using executed GT segment.
            exec_end = s + spec.exec_h
            if exec_end <= total_future_len:
                exec_segment = agents_future[:, s:exec_end, :5]
                exec_segment_t = torch.from_numpy(exec_segment).unsqueeze(0).to(
                    device=device, dtype=working_batch["agents_history"].dtype
                )
                working_batch = update_batch_for_training(
                    batch=working_batch,
                    predicted_segment=exec_segment_t,
                )

        agent_ids = sample.get("agents_id", None)
        if agent_ids is not None:
            agent_ids = np.asarray(agent_ids)[:agents_len]

        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "agent_ids": agent_ids,
            "agent_latents_segments": np.stack(z_segments, axis=0).astype(np.float32),  # [S,A,Za]
            "num_segments": int(len(z_segments)),
            "latent_index_mode": "window",
            "exec_h": int(spec.exec_h),
            "pred_h": int(spec.pred_h),
            "future_len": int(spec.state_len),
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved latents to: {out_latent_dir}")


@torch.no_grad()
def main(
    *,
    vbd_dir: str,
    out_latent_dir: str,
    ckpt_path: str,
    cfg_path: str,
    agents_len: int,
    exec_h: int,
    pred_h: int,
    sample_latent: bool,
    overwrite: bool,
):
    spec = WindowSpec(exec_h=exec_h, pred_h=pred_h)

    # Load BehaviorVAE config (prefer ckpt_dir/config.yaml when cfg=auto).
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
    if cfg_path is None or str(cfg_path).lower() == "auto":
        cfg = load_config(ckpt_cfg_path)
    else:
        cfg = load_config(cfg_path)

    # Load anchors needed by the VBD context encoder inside BehaviorVAE.
    anchor_path = cfg.get("anchor_path")
    if not anchor_path:
        raise ValueError("BehaviorVAE config must contain 'anchor_path' for context encoder inputs")
    anchor_path = Path(anchor_path).expanduser()
    if not anchor_path.is_absolute():
        cfg_ref_path = Path(ckpt_cfg_path if cfg_path is None or str(cfg_path).lower() == "auto" else cfg_path)
        candidates = [
            cfg_ref_path.parent / anchor_path,
            _VBD_ROOT / anchor_path,
            Path.cwd() / anchor_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                anchor_path = candidate
                break
    with open(anchor_path, "rb") as f:
        anchor_dict = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorVAE(cfg=cfg).to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)

    _generate_latents_for_dir(
        vbd_dir=vbd_dir,
        out_latent_dir=out_latent_dir,
        model=model,
        anchor_dict=anchor_dict,
        device=device,
        spec=spec,
        agents_len=agents_len,
        sample_latent=sample_latent,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root with training/testing vbd splits")
    parser.add_argument("--split", type=str, default="all", choices=["training", "testing", "all"])
    parser.add_argument("--latent_dir_name", type=str, default="latents_behaviorvae")
    parser.add_argument("--vbd_dir", type=str, default=None, help="Folder of VBD scenario .pkl files")
    parser.add_argument("--out_latent_dir", type=str, default=None, help="Output folder for scenario_<id>_latents.pkl")
    parser.add_argument("--ckpt", type=str, required=True, help="BehaviorVAE checkpoint (.ckpt)")
    parser.add_argument("--cfg", type=str, default="auto", help="BehaviorVAE config yaml or 'auto' to use <ckpt_dir>/config.yaml")
    parser.add_argument("--agents_len", type=int, default=32)
    parser.add_argument("--exec_h", type=int, default=11)
    parser.add_argument("--pred_h", type=int, default=40)
    parser.add_argument("--sample_latent", action="store_true", help="sample z instead of using mean")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset_root:
        splits = ["training", "testing"] if args.split == "all" else [args.split]
        for split in splits:
            vbd_dir = os.path.join(args.dataset_root, split, "vbd")
            out_latent_dir = os.path.join(args.dataset_root, split, args.latent_dir_name)
            main(
                vbd_dir=vbd_dir,
                out_latent_dir=out_latent_dir,
                ckpt_path=args.ckpt,
                cfg_path=args.cfg,
                agents_len=args.agents_len,
                exec_h=args.exec_h,
                pred_h=args.pred_h,
                sample_latent=bool(args.sample_latent),
                overwrite=bool(args.overwrite),
            )
    else:
        if not args.vbd_dir or not args.out_latent_dir:
            raise SystemExit("Must provide --vbd_dir and --out_latent_dir when --dataset_root is not set.")
        main(
            vbd_dir=args.vbd_dir,
            out_latent_dir=args.out_latent_dir,
            ckpt_path=args.ckpt,
            cfg_path=args.cfg,
            agents_len=args.agents_len,
            exec_h=args.exec_h,
            pred_h=args.pred_h,
            sample_latent=bool(args.sample_latent),
            overwrite=bool(args.overwrite),
        )
