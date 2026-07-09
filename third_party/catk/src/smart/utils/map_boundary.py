from typing import Dict, Tuple

import torch
from torch import Tensor


def compute_map_boundary(
    traj_pos: Tensor,
    margin: float = 0.0,
) -> Tensor:
    xy = traj_pos[..., :2].reshape(-1, 2)
    valid = torch.isfinite(xy).all(dim=-1)
    xy = xy[valid]
    if xy.numel() == 0:
        return torch.tensor(
            [-6e4, 6e4, -6e4, 6e4],
            dtype=traj_pos.dtype,
            device=traj_pos.device,
        )
    xmin = xy[:, 0].min() - margin
    xmax = xy[:, 0].max() + margin
    ymin = xy[:, 1].min() - margin
    ymax = xy[:, 1].max() + margin
    return torch.stack([xmin, xmax, ymin, ymax], dim=0)


def compute_batch_boundaries(
    map_traj_pos: Tensor,
    map_batch: Tensor,
    num_graphs: int,
    margin: float = 0.0,
) -> Tensor:
    boundaries = []
    for i in range(num_graphs):
        mask = map_batch == i
        boundaries.append(compute_map_boundary(map_traj_pos[mask], margin=margin))
    return torch.stack(boundaries, dim=0)


def points_in_boundary(points: Tensor, boundaries: Tensor) -> Tensor:
    xmin, xmax, ymin, ymax = boundaries.unbind(dim=-1)
    while xmin.dim() < points.dim() - 1:
        xmin = xmin.unsqueeze(-1)
        xmax = xmax.unsqueeze(-1)
        ymin = ymin.unsqueeze(-1)
        ymax = ymax.unsqueeze(-1)
    x = points[..., 0]
    y = points[..., 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def truncate_sequence_by_boundary(
    valid_mask: Tensor,
    positions: Tensor,
    boundary: Tensor,
    start_step: int = 0,
) -> Tensor:
    valid_mask = valid_mask.clone()
    inside = points_in_boundary(positions, boundary.view(1, 4).expand(positions.shape[0], -1))
    inside = inside & valid_mask
    if start_step > 0:
        inside[:, :start_step] = valid_mask[:, :start_step]

    for i in range(valid_mask.shape[0]):
        invalid_steps = torch.where(valid_mask[i] & ~inside[i])[0]
        if len(invalid_steps) == 0:
            continue
        first_invalid = int(invalid_steps[0].item())
        valid_mask[i, first_invalid:] = False
    return valid_mask


def truncate_batch_by_boundary(
    valid_mask: Tensor,
    positions: Tensor,
    batch: Tensor,
    boundaries: Tensor,
    start_step: int = 0,
) -> Tensor:
    valid_mask = valid_mask.clone()
    for i in range(boundaries.shape[0]):
        mask = batch == i
        if not mask.any():
            continue
        valid_mask[mask] = truncate_sequence_by_boundary(
            valid_mask[mask],
            positions[mask],
            boundaries[i],
            start_step=start_step,
        )
    return valid_mask


def freeze_trajectory_after_invalid(
    valid_mask: Tensor,
    traj: Tensor,
    head: Tensor,
) -> Tuple[Tensor, Tensor]:
    traj = traj.clone()
    head = head.clone()
    for i in range(valid_mask.shape[0]):
        invalid_steps = torch.where(~valid_mask[i])[0]
        if len(invalid_steps) == 0:
            continue
        first_invalid = int(invalid_steps[0].item())
        if first_invalid == 0:
            continue
        traj[i, first_invalid:] = traj[i, first_invalid - 1].unsqueeze(0)
        head[i, first_invalid:] = head[i, first_invalid - 1]
    return traj, head


def freeze_rollouts_after_invalid(
    valid_mask: Tensor,
    traj: Tensor,
    head: Tensor,
) -> Tuple[Tensor, Tensor]:
    traj = traj.clone()
    head = head.clone()
    for i in range(valid_mask.shape[0]):
        for j in range(valid_mask.shape[1]):
            invalid_steps = torch.where(~valid_mask[i, j])[0]
            if len(invalid_steps) == 0:
                continue
            first_invalid = int(invalid_steps[0].item())
            if first_invalid == 0:
                continue
            traj[i, j, first_invalid:] = traj[i, j, first_invalid - 1].unsqueeze(0)
            head[i, j, first_invalid:] = head[i, j, first_invalid - 1]
    return traj, head


def make_boundary_config(
    enable: bool = False,
    margin: float = 3.0,
    gt_start_step: int = 11,
) -> Dict[str, float]:
    return {
        "enable": bool(enable),
        "margin": float(margin),
        "gt_start_step": int(gt_start_step),
    }
