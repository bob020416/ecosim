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

import os

from src.smart.metrics.cross_entropy import CrossEntropy
from src.smart.metrics.ego_nll import EgoNLL
from src.smart.metrics.gmm_ade import GMMADE
from src.smart.metrics.min_ade import minADE
from src.smart.metrics.next_token_cls import TokenCls
from src.smart.metrics.wosac_submission import WOSACSubmission

_SKIP_TF = os.environ.get("CATK_SKIP_TF", "") == "1"
if _SKIP_TF:
    class WOSACMetrics:  # noqa: N801 - keep public API name
        def __init__(self, *args, **kwargs) -> None:
            pass

        def update(self, *args, **kwargs) -> None:
            pass

        def compute(self):
            return {}

        def reset(self) -> None:
            pass
else:
    from src.smart.metrics.wosac_metrics import WOSACMetrics
