from pathlib import Path
from typing import Optional, Tuple

import pickle
import torch


def load_latent_payload(latent_dir: Path, scenario_id: str) -> dict:
    latent_path = latent_dir / f"scenario_{scenario_id}_latents.pkl"
    if not latent_path.exists():
        raise RuntimeError(f"Latent file not found: {latent_path}")
    with open(latent_path, "rb") as f:
        payload = pickle.load(f)
    return payload


def align_latents_to_agents(
    payload: dict,
    agent_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    agent_latents = payload.get("agent_latents")
    if agent_latents is None:
        raise ValueError("agent_latents missing in latent payload")
    agent_latents = torch.as_tensor(agent_latents, dtype=torch.float32)
    if agent_latents.dim() != 3:
        raise ValueError(
            f"agent_latents must be [S,A,Za], got {agent_latents.shape}"
        )
    agent_latents = agent_latents.permute(1, 0, 2).contiguous()  # [A,S,Za]

    agent_ids_cpu = agent_ids.detach().cpu()
    agent_ids_list = agent_ids_cpu.tolist()
    n_agent = len(agent_ids_list)

    latent_ids = payload.get("agent_ids")
    if latent_ids is None:
        if agent_latents.shape[0] != n_agent:
            raise ValueError(
                f"latent agent_ids missing and count mismatch ({agent_latents.shape[0]} vs {n_agent})"
            )
        latents = agent_latents
        latent_mask = torch.ones(n_agent, dtype=torch.bool)
    else:
        latent_ids_list = (
            latent_ids.tolist() if hasattr(latent_ids, "tolist") else list(latent_ids)
        )
        idx_map = {int(aid): idx for idx, aid in enumerate(latent_ids_list)}
        latents = torch.zeros(
            (n_agent, agent_latents.shape[1], agent_latents.shape[2]),
            dtype=agent_latents.dtype,
        )
        latent_mask = torch.zeros(n_agent, dtype=torch.bool)
        for i, aid in enumerate(agent_ids_list):
            idx = idx_map.get(int(aid))
            if idx is None:
                continue
            latents[i] = agent_latents[idx]
            latent_mask[i] = True

    scene_latents = payload.get("scene_latents")
    if scene_latents is not None:
        scene_latents = torch.as_tensor(scene_latents, dtype=torch.float32)

    return (
        latents.to(device),
        latent_mask.to(device),
        scene_latents.to(device) if scene_latents is not None else None,
    )


__all__ = ["load_latent_payload", "align_latents_to_agents"]
