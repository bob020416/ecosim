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

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

import lightning as L
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.smart.datasets import MultiDataset
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils import cal_polygon_contour, transform_to_local, wrap_angle


def default_width_length(agent_type: int) -> torch.Tensor:
    if agent_type == 0:
        return torch.tensor([2.0, 4.8], dtype=torch.float32)
    if agent_type == 1:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    return torch.tensor([1.0, 2.0], dtype=torch.float32)


def Kdisk_cluster(
    X,  # [n_trajs, 4, 2], bbox of the last point of the segment
    N,  # int
    tol,  # float
    a_pos,  # [n_trajs, 6, 3], the complete segment
    cal_mean_heading=True,
):
    n_total = X.shape[0]
    ret_traj_list = []
    fallback_traj = a_pos[:1].clone()

    for i in range(N):
        if X.shape[0] == 0:
            print(f"{i=}, remain=0.00%, exhausted candidates; repeating last prototype")
            ret_traj_list.append(fallback_traj.clone())
            continue
        if i == 0:
            choice_index = 0  # always include [0, 0, 0]
        else:
            choice_index = torch.randint(0, X.shape[0], (1,)).item()
        x0 = X[choice_index]
        # res_mask = torch.sum((X - x0) ** 2, dim=[1, 2]) / 4.0 > (tol**2)
        res_mask = torch.norm(X - x0, dim=-1).mean(-1) > tol
        if cal_mean_heading:
            ret_traj = a_pos[~res_mask].mean(0, keepdim=True)
        else:
            ret_traj = a_pos[[choice_index]]
        fallback_traj = ret_traj
        X = X[res_mask]
        a_pos = a_pos[res_mask]
        ret_traj_list.append(ret_traj)

        remain = X.shape[0] * 100.0 / n_total
        n_inside = (~res_mask).sum().item()
        print(f"{i=}, {remain=:.2f}%, {n_inside=}")

    return torch.cat(ret_traj_list, dim=0)  # [N, 6, 3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build agent token vocabulary from SMART-format cache.")
    parser.add_argument(
        "--data-cache-path",
        type=Path,
        default=Path("./cache/SMART"),
        help="SMART-format cache root containing split folders such as training/train/valid.",
    )
    parser.add_argument(
        "--input-split",
        type=str,
        default="training",
        help="Cache split directory name under data-cache-path.",
    )
    parser.add_argument(
        "--out-file-name",
        type=str,
        default="agent_vocab_555_s2.pkl",
        help="Output vocabulary filename written next to this script.",
    )
    parser.add_argument(
        "--traj-cache-file",
        type=str,
        default="kdisk_trajs.pkl",
        help="Intermediate trajectory cache filename stored under data-cache-path.",
    )
    parser.add_argument(
        "--load-data-from-file",
        action="store_true",
        help="Load the intermediate trajectory cache instead of rebuilding it.",
    )
    args = parser.parse_args()

    L.seed_everything(seed=2, workers=True)
    n_trajs = 2048 * 100  # 2e5
    load_data_from_file = args.load_data_from_file
    data_cache_path = args.data_cache_path
    out_file_name = args.out_file_name
    tol_dist = [0.05, 0.05, 0.05]  # veh, ped, cyc

    # ! don't change these params
    shift = 5  # motion token time dimension
    num_cluster = 2048  # vocabulary size
    n_step = 91
    data_file_path = data_cache_path / args.traj_cache_file
    if load_data_from_file:
        with open(data_file_path, "rb") as f:
            data = pickle.load(f)
    else:
        default_shapes = [
            torch.tensor([[2.0, 4.8]], dtype=torch.float32),  # veh
            torch.tensor([[1.0, 1.0]], dtype=torch.float32),  # ped
            torch.tensor([[1.0, 2.0]], dtype=torch.float32),  # cyc
        ]
        trajs = [
            torch.zeros([1, 6, 3], dtype=torch.float32),  # veh
            torch.zeros([1, 6, 3], dtype=torch.float32),  # ped
            torch.zeros([1, 6, 3], dtype=torch.float32),  # cyc
        ]
        shapes = [x.clone() for x in default_shapes]
        dataloader = DataLoader(
            dataset=MultiDataset(
                raw_dir=data_cache_path / args.input_split,
                transform=lambda x: HeteroData(x),
            ),
            batch_size=8,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

        with tqdm(
            total=len(dataloader),
            desc=f"n_trajs={n_trajs}",
            postfix={"n_veh": 0, "n_ped": 0, "n_cyc": 0},
        ) as pbar:

            for data in dataloader:
                valid_mask = data["agent"]["valid_mask"]
                data["agent"]["heading"] = TokenProcessor._clean_heading(
                    valid_mask, data["agent"]["heading"]
                )

                for i_ag in range(valid_mask.shape[0]):
                    if valid_mask[i_ag, :].sum() < 30:
                        continue
                    for t in range(0, n_step - shift, shift):
                        if valid_mask[i_ag, t] and valid_mask[i_ag, t + shift]:
                            _type = data["agent"]["type"][i_ag]
                            if trajs[_type].shape[0] < n_trajs:
                                pos = data["agent"]["position"][
                                    i_ag, t : t + shift + 1, :2
                                ]
                                head = data["agent"]["heading"][i_ag, t : t + shift + 1]
                                shape = data["agent"]["shape"][i_ag, [1, 0]].float()
                                if (shape <= 0).any():
                                    shape = default_width_length(int(_type)).to(shape.device)
                                pos, head = transform_to_local(
                                    pos_global=pos.unsqueeze(0),  # [1, 6, 2]
                                    head_global=head.unsqueeze(0),  # [1, 6]
                                    pos_now=pos[[0]],  # [1, 2]
                                    head_now=head[[0]],  # [1]
                                )
                                head = wrap_angle(head)
                                to_add = torch.cat([pos, head.unsqueeze(-1)], dim=-1)

                                if not (
                                    (
                                        (trajs[_type] - to_add).abs().sum([1, 2]) < 1e-2
                                    ).any()
                                ):
                                    trajs[_type] = torch.cat(
                                        [trajs[_type], to_add], dim=0
                                    )
                                    shapes[_type] = torch.cat(
                                        [shapes[_type], shape.unsqueeze(0)], dim=0
                                    )
                pbar.update(1)
                pbar.set_postfix(
                    n_veh=trajs[0].shape[0],
                    n_ped=trajs[1].shape[0],
                    n_cyc=trajs[2].shape[0],
                )
                if (
                    trajs[0].shape[0] == n_trajs
                    and trajs[1].shape[0] == n_trajs
                    and trajs[2].shape[0] == n_trajs
                ):
                    break

        # [n_trajs, shift+1, [relative_x, relative_y, relative_theta]]
        data = {
            "veh": {"traj": trajs[0], "shape": shapes[0]},
            "ped": {"traj": trajs[1], "shape": shapes[1]},
            "cyc": {"traj": trajs[2], "shape": shapes[2]},
        }

        with open(data_file_path, "wb") as f:
            pickle.dump(data, f)

    res = {"token_all": {}, "traj_all": {}, "shape_mean": {}}

    for k, v in data.items():
        if isinstance(v, dict):
            traj = v["traj"].float()
            width_length = v["shape"].float()
        else:
            traj = v.float()
            if k == "veh":
                width_length = torch.tensor([[2.0, 4.8]], dtype=torch.float32)
            elif k == "ped":
                width_length = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
            elif k == "cyc":
                width_length = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        contour = cal_polygon_contour(
            pos=traj[:, -1:, :2],
            head=traj[:, -1:, 2],
            width_length=width_length[:, None, :],
        ).squeeze(1)  # [n_trajs, 4, 2]

        if k == "veh":
            tol = tol_dist[0]
        elif k == "ped":
            tol = tol_dist[1]
        elif k == "cyc":
            tol = tol_dist[2]
        print(k, tol)
        ret_traj = Kdisk_cluster(X=contour, N=num_cluster, tol=tol, a_pos=traj)
        ret_traj[:, :, -1] = wrap_angle(ret_traj[:, :, -1])

        mean_width_length = width_length.mean(0).view(1, 1, 2)
        contour = cal_polygon_contour(
            pos=ret_traj[:, :, :2],  # [N, 6, 2]
            head=ret_traj[:, :, 2],  # [N, 6]
            width_length=mean_width_length,
        )
        res["token_all"][k] = contour.numpy()
        res["traj_all"][k] = ret_traj.numpy()
        res["shape_mean"][k] = mean_width_length.squeeze(0).numpy()

    with open(Path(__file__).resolve().parent / out_file_name, "wb") as f:
        pickle.dump(res, f)
