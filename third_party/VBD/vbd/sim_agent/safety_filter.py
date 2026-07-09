import torch
import numpy as np
from typing import Tuple, Dict, List

# Helper function to calculate agent dimensions for collision checking
def get_agent_radii(agent_dims: np.ndarray) -> np.ndarray:
    """
    Calculates the radius of agents assuming they are approximated by circles.
    The radius is based on half the diagonal of the bounding box.
    agent_dims: np.ndarray of shape [A, 2] (Length, Width)
    Returns: radii: np.ndarray of shape [A]
    """
    # Calculate radius based on half the diagonal of the bounding box
    # r = sqrt(L^2 + W^2) / 2
    radii = np.linalg.norm(agent_dims, axis=-1) / 2.0
    return radii

def select_safest_trajectory(
    pred_trajs: torch.Tensor,  # Shape: [M, B, A, T, D]
    agent_dims: torch.Tensor,  # Shape: [B, A, 2] (Length, Width) for each agent
    agent_valid_mask: torch.Tensor = None, # Shape: [B, A] (Boolean or 0/1)
    # Add other potential cost functions here if needed later (e.g., offroad, smoothness)
) -> torch.Tensor:
    """
    Selects the safest trajectory from M candidates based on collision count.

    Each agent is approximated by a single circle with radius equal to half the
    diagonal of the bounding box.
    """
    M, B, A, T, D = pred_trajs.shape
    device = pred_trajs.device

    if agent_dims.device != device:
        agent_dims = agent_dims.to(device)
    if agent_valid_mask is not None and agent_valid_mask.device != device:
        agent_valid_mask = agent_valid_mask.to(device)

    radii = torch.norm(agent_dims, dim=-1) / 2.0  # [B, A]
    radii_expanded = radii.unsqueeze(0).unsqueeze(2)  # [1, B, A, 1]
    pos = pred_trajs[..., :2]  # [M, B, A, T, 2]

    collision_cost = torch.zeros(M, B, device=device)
    r_sum = radii_expanded + radii_expanded.transpose(2, 3)  # [1, B, A, A]
    r_sum = r_sum.unsqueeze(-1)  # [1, B, A, A, 1]
    mask_self = torch.eye(A, device=device).view(1, 1, A, A, 1)

    if agent_valid_mask is not None:
        v = agent_valid_mask.unsqueeze(-1)
        valid_pair = (v * v.transpose(1, 2)).unsqueeze(0).unsqueeze(-1)  # [1, B, A, A, 1]
    else:
        valid_pair = torch.ones(1, B, A, A, 1, device=device)

    for m in range(M):
        p = pos[m]  # [B, A, T, 2]
        dist = torch.norm(p.unsqueeze(2) - p.unsqueeze(1), dim=-1)  # [B, A, A, T]
        is_coll = (dist < r_sum[0]) & (mask_self[0] == 0) & (valid_pair[0] > 0)
        collision_cost[m] = is_coll.float().sum(dim=(1, 2, 3))

    best_indices = torch.argmin(collision_cost, dim=0)  # [B]
    best_trajectories = pred_trajs[best_indices, torch.arange(B, device=device)]
    return best_trajectories
