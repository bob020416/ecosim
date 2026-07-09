#!/usr/bin/env python
"""Create CATK-aligned latent pairs using BehaviorVAE (next 4s windows)."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar dependency
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure VBD modules are importable in the ECoSim release layout:
# third_party/catk and third_party/VBD are sibling directories.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VBD_ROOT = _PROJECT_ROOT.parent / "VBD"
if _VBD_ROOT.exists() and str(_VBD_ROOT) not in sys.path:
    sys.path.insert(0, str(_VBD_ROOT))

from vbd.model.behavior_vae import BehaviorVAE
from vbd.model.model_utils import update_batch_for_training

"""
cd third_party/catk
python scripts/latent/create_catk_latent_pairs.py \
  --dataset_root ../VBD/data/waymovalid-scenarionet \
  --ckpt ../../checkpoints/vbd/behaviorvae.ckpt \
  --cfg ../VBD/config/BehaviorVAE_full.yaml \
  --latent_dir_name latents_catk_4s_wcontext \
  --stride 5 \
  --pred_h 40 \
  --num_steps 16

Or specify one VBD scenario directory:

python scripts/latent/create_catk_latent_pairs.py \
  --vbd_dir ../VBD/data/waymovalid-scenarionet/training/vbd \
  --out_latent_dir ../VBD/data/waymovalid-scenarionet/training/latents_catk_4s_wcontext \
  --ckpt ../../checkpoints/vbd/behaviorvae.ckpt \
  --cfg ../VBD/config/BehaviorVAE_full.yaml \
  --stride 5 --pred_h 40 --num_steps 16
"""

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


@dataclass
class WindowSpec:
    stride: int
    pred_h: int

    @property
    def future_len(self) -> int:
        return self.pred_h + 1


def _compute_anchors(agents_type: np.ndarray, anchors_dict: Dict[str, Any]) -> np.ndarray:
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


def _to_torch_batch(sample: Dict[str, Any], agents_future_window: np.ndarray) -> Dict[str, torch.Tensor]:
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
    out = {k: v.unsqueeze(0) for k, v in out.items()}
    return out


def _slice_window(
    agents_future: np.ndarray, start: int, future_len: int
) -> np.ndarray:
    total_len = agents_future.shape[1]
    window = np.zeros((agents_future.shape[0], future_len, agents_future.shape[2]), dtype=agents_future.dtype)
    if start >= total_len:
        return window
    end = min(total_len, start + future_len)
    window[:, : end - start, :] = agents_future[:, start:end, :]
    return window


def _iter_starts(num_steps: int, stride: int) -> List[int]:
    return [i * stride for i in range(num_steps)]


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
    num_steps: int,
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
            scenario_id = os.path.splitext(fname)[0].replace("scenario_", "")

        out_path = os.path.join(out_latent_dir, f"scenario_{scenario_id}_latents.pkl")
        if os.path.exists(out_path) and not overwrite:
            continue

        agents_future = sample["agents_future"]  # [N,T,5]

        if "anchors" not in sample:
            sample = dict(sample)
            sample["anchors"] = _compute_anchors(sample["agents_type"], anchor_dict)

        starts = _iter_starts(num_steps=num_steps, stride=spec.stride)
        init_window = _slice_window(agents_future, 0, spec.future_len)
        working_batch = _to_torch_batch(sample, agents_future_window=init_window)
        working_batch = {k: v.to(device) for k, v in working_batch.items()}

        z_agents: list[np.ndarray] = []
        z_scenes: list[np.ndarray] = []

        for s in tqdm(starts, desc=f"{scenario_id}", unit="step", leave=False):
            window = _slice_window(agents_future, s, spec.future_len)
            working_batch["agents_future"] = (
                torch.from_numpy(window)
                .unsqueeze(0)
                .to(device=device, dtype=working_batch["agents_history"].dtype)
            )

            out = model(working_batch, sample_latent=sample_latent)
            z_agent = out.z_agent[0, :agents_len].detach().cpu().numpy()
            z_agents.append(z_agent)
            if out.z_scene is not None:
                z_scenes.append(out.z_scene[0].detach().cpu().numpy())

            exec_segment = _slice_window(agents_future, s, spec.stride)
            exec_segment_t = (
                torch.from_numpy(exec_segment)
                .unsqueeze(0)
                .to(device=device, dtype=working_batch["agents_history"].dtype)
            )
            working_batch = update_batch_for_training(
                batch=working_batch,
                predicted_segment=exec_segment_t,
            )

        # Prefer Waymo track ids if available; fall back to index ids.
        agent_ids = sample.get("agent_track_id", None)
        if agent_ids is None:
            agent_ids = sample.get("agents_id", None)
        if agent_ids is not None:
            agent_ids = np.asarray(agent_ids)[:agents_len]

        payload: Dict[str, Any] = {
            "scenario_id": scenario_id,
            "agent_ids": agent_ids,
            "agent_latents": np.stack(z_agents, axis=0).astype(np.float32),  # [S,A,Za]
            "scene_latents": np.stack(z_scenes, axis=0).astype(np.float32) if z_scenes else None,
            "num_steps": int(len(z_agents)),
            "stride": int(spec.stride),
            "pred_h": int(spec.pred_h),
            "future_len": int(spec.future_len),
            "source_hz": 10,
            "catk_steps": int(num_steps),
            "start_indices": np.asarray(starts, dtype=np.int32),
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
    stride: int,
    pred_h: int,
    num_steps: int,
    sample_latent: bool,
    overwrite: bool,
):
    spec = WindowSpec(stride=stride, pred_h=pred_h)

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
    if cfg_path is None or str(cfg_path).lower() == "auto":
        cfg = load_config(ckpt_cfg_path)
    else:
        cfg = load_config(cfg_path)

    anchor_path = cfg.get("anchor_path")
    if not anchor_path:
        raise ValueError("BehaviorVAE config must contain 'anchor_path' for context encoder inputs")
    anchor_path = Path(anchor_path)
    if not anchor_path.is_absolute():
        cfg_dir = Path(ckpt_cfg_path).parent
        candidate = cfg_dir / anchor_path
        if candidate.exists():
            anchor_path = candidate
        elif _VBD_ROOT.exists():
            anchor_path = _VBD_ROOT / anchor_path
    with open(anchor_path.as_posix(), "rb") as f:
        anchor_dict = pickle.load(f)

    if agents_len <= 0:
        agents_len = int(cfg.get("agents_len", 64))

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
        num_steps=num_steps,
        sample_latent=sample_latent,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root with VBD splits")
    parser.add_argument("--split", type=str, default="all", choices=["training", "testing", "all"])
    parser.add_argument("--latent_dir_name", type=str, default="latents_catk_4s")
    parser.add_argument("--vbd_dir", type=str, default=None, help="Folder of VBD scenario .pkl files")
    parser.add_argument("--out_latent_dir", type=str, default=None, help="Output folder for scenario_<id>_latents.pkl")
    parser.add_argument("--ckpt", type=str, required=True, help="BehaviorVAE checkpoint (.ckpt)")
    parser.add_argument("--cfg", type=str, default="auto", help="BehaviorVAE config yaml or 'auto' to use <ckpt_dir>/config.yaml")
    parser.add_argument("--agents_len", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5, help="Stride in 10Hz steps (5=0.5s)")
    parser.add_argument("--pred_h", type=int, default=40, help="Prediction horizon in 10Hz steps (40=4s)")
    parser.add_argument("--num_steps", type=int, default=16, help="CATK steps to align (default 16)")
    parser.add_argument("--sample_latent", action="store_true", help="Sample z instead of using mean")
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
                stride=args.stride,
                pred_h=args.pred_h,
                num_steps=args.num_steps,
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
            stride=args.stride,
            pred_h=args.pred_h,
            num_steps=args.num_steps,
            sample_latent=bool(args.sample_latent),
            overwrite=bool(args.overwrite),
        )
