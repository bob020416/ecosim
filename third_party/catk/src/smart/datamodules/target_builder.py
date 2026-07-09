# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from src.smart.utils import compute_map_boundary, truncate_sequence_by_boundary


class WaymoTargetBuilderTrain(BaseTransform):
    def __init__(
        self,
        max_num: int,
        enforce_map_boundary: bool = False,
        map_boundary_margin: float = 3.0,
        map_boundary_gt_start_step: int = 11,
    ) -> None:
        super(WaymoTargetBuilderTrain, self).__init__()
        self.step_current = 10
        self.max_num = max_num
        self.enforce_map_boundary = enforce_map_boundary
        self.map_boundary_margin = map_boundary_margin
        self.map_boundary_gt_start_step = map_boundary_gt_start_step

    def forward(self, data) -> HeteroData:
        if self.enforce_map_boundary:
            boundary = compute_map_boundary(
                data["map_save"]["traj_pos"], margin=self.map_boundary_margin
            )
            data["agent"]["valid_mask"] = truncate_sequence_by_boundary(
                data["agent"]["valid_mask"],
                data["agent"]["position"][..., :2],
                boundary,
                start_step=self.map_boundary_gt_start_step,
            )

        pos = data["agent"]["position"]
        av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        distance = torch.norm(pos - pos[av_index], dim=-1)

        # we do not believe the perception out of range of 150 meters
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

        # we do not predict vehicle too far away from ego car
        role_train_mask = data["agent"]["role"].any(-1)
        extra_train_mask = (distance[:, self.step_current] < 100) & (
            data["agent"]["valid_mask"][:, self.step_current + 1 :].sum(-1) >= 5
        )

        train_mask = extra_train_mask | role_train_mask
        if train_mask.sum() > self.max_num:  # too many vehicle
            _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
            selected_indices = _indices[
                torch.randperm(_indices.size(0))[: self.max_num - role_train_mask.sum()]
            ]
            data["agent"]["train_mask"] = role_train_mask
            data["agent"]["train_mask"][selected_indices] = True
        else:
            data["agent"]["train_mask"] = train_mask  # [n_agent]

        return HeteroData(data)

    def __call__(self, data) -> HeteroData:
        return self.forward(data)


class WaymoTargetBuilderVal(BaseTransform):
    def __init__(
        self,
        enforce_map_boundary: bool = False,
        map_boundary_margin: float = 3.0,
        map_boundary_gt_start_step: int = 11,
    ) -> None:
        super(WaymoTargetBuilderVal, self).__init__()
        self.enforce_map_boundary = enforce_map_boundary
        self.map_boundary_margin = map_boundary_margin
        self.map_boundary_gt_start_step = map_boundary_gt_start_step

    def forward(self, data) -> HeteroData:
        if self.enforce_map_boundary:
            boundary = compute_map_boundary(
                data["map_save"]["traj_pos"], margin=self.map_boundary_margin
            )
            data["agent"]["valid_mask"] = truncate_sequence_by_boundary(
                data["agent"]["valid_mask"],
                data["agent"]["position"][..., :2],
                boundary,
                start_step=self.map_boundary_gt_start_step,
            )
        return HeteroData(data)

    def __call__(self, data) -> HeteroData:
        return self.forward(data)
