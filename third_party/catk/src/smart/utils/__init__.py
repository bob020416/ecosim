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

from src.smart.utils.geometry import angle_between_2d_vectors, wrap_angle
from src.smart.utils.map_boundary import (
    compute_batch_boundaries,
    compute_map_boundary,
    freeze_rollouts_after_invalid,
    freeze_trajectory_after_invalid,
    truncate_batch_by_boundary,
    truncate_sequence_by_boundary,
)
from src.smart.utils.rollout import (
    cal_polygon_contour,
    sample_next_gmm_traj,
    sample_next_token_traj,
    transform_to_global,
    transform_to_local,
)
from src.smart.utils.weight_init import weight_init
