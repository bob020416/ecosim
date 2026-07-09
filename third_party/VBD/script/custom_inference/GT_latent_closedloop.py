import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

import utils_closed_loop as utils_cl
import utils as utils_ol

"""Closed-loop or open-loop VBD inference with per-scenario GT latent conditioning."""

_LATENT_CACHE: Dict[str, Dict] = {}


def _normalize_scenario_id(raw_id) -> str:
    if raw_id is None:
        raise ValueError("Missing scenario_id in batch; cannot load latents.")
    if isinstance(raw_id, (bytes, bytearray)):
        return raw_id.decode("utf-8")
    return str(raw_id)


def _load_latent_segments(latent_dir: str, scenario_id: str) -> Dict:
    if scenario_id in _LATENT_CACHE:
        return _LATENT_CACHE[scenario_id]

    candidate = os.path.join(latent_dir, f"scenario_{scenario_id}_latents.pkl")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Latent file not found: {candidate}")

    with open(candidate, "rb") as f:
        data = pickle.load(f)

    agent_segments = data.get("agent_latents_segments")
    if agent_segments is None:
        agent_latents = data.get("agent_latents")
        if agent_latents is None:
            raise KeyError("Latent file missing agent_latents_segments/agent_latents")
        agent_segments = np.expand_dims(agent_latents, axis=0)

    segment_latents: List[Dict[str, np.ndarray]] = []
    for seg in agent_segments:
        scenario_latents = np.zeros((seg.shape[0], 1), dtype=np.float32)
        segment_latents.append(
            {
                "agent_latents": seg.astype(np.float32),
                "scenario_latents": scenario_latents,
            }
        )

    payload = {
        "segment_latents": segment_latents,
        "num_segments": len(segment_latents),
        "agent_latents": segment_latents[0]["agent_latents"],
        "scenario_latents": segment_latents[0]["scenario_latents"],
        "latent_index_mode": data.get("latent_index_mode", "window"),
        "latent_segment_len": int(data.get("latent_segment_len", data.get("exec_h", 11))),
    }

    _LATENT_CACHE[scenario_id] = payload
    return payload


def _gt_latent(batch: dict) -> Tuple[Dict, List[int]]:
    scenario_id = _normalize_scenario_id(batch.get("scenario_id"))
    latents = _load_latent_segments(_gt_latent.latent_dir, scenario_id)

    agent_latents = latents["agent_latents"]
    active = np.where(np.any(agent_latents != 0.0, axis=-1))[0].tolist()
    return latents, active


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_path", type=str, required=True, help="Directory containing PKL files")
    parser.add_argument("--latent_path", type=str, required=True, help="Directory containing latent PKL files")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ctrl_ckpt_path", type=str, default=None)
    parser.add_argument("--test_mode", type=str, default="diffusion", choices=["diffusion", "prior"])
    parser.add_argument("--save_simulation", action="store_true")
    parser.add_argument("--plot_control_samples", action="store_true")
    parser.add_argument("--use_training_format", action="store_true", default=True)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="test_closedloop_gt_latent")
    parser.add_argument("--only_controlnet", action="store_true", help="Only run ControlNet (skip baseline) for speed.")
    parser.add_argument("--closedloop", action="store_true")
    parser.add_argument("--horizon_steps", type=int, default=11)
    parser.add_argument("--total_steps", type=int, default=80)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=32)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _gt_latent.latent_dir = args.latent_path

    if args.closedloop:
        print("\n🔄 Running CLOSED-LOOP simulation with GT latent conditioning")
        utils_cl.run_closedloop_simulation(args, _gt_latent, sketch_name="GT Latent")
    else:
        print("\n➡️  Running OPEN-LOOP simulation with GT latent conditioning")
        utils_ol.run_simulation(args, _gt_latent, sketch_name="GT Latent")


if __name__ == "__main__":
    main()
