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

from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader

from src.smart.datasets import MultiDataset

from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        val_tfrecords_splitted: str,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_max_num: int,
        train_latent_dir: Optional[str] = None,
        val_latent_dir: Optional[str] = None,
        test_latent_dir: Optional[str] = None,
        text_prompt_root: Optional[str] = None,
        text_mapping_path: Optional[str] = None,
        text_prompt_subdir: str = "70b_0.8_0.9_action_prompt_v2",
        text_split: str = "auto",
        text_mode: str = "full",
        text_only_mapped: bool = False,
        text_key: str = "text_prompt",
        text_mask_key: str = "text_prompt_mask",
        tag_prompt_root: Optional[str] = None,
        tag_prompt_subdir: str = "auto",
        tag_time_scale: float = 0.1,
        tag_sentence_style: str = "ordered",
        enforce_map_boundary: bool = False,
        map_boundary_margin: float = 3.0,
        map_boundary_gt_start_step: int = 11,
    ) -> None:
        super(MultiDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted
        self.train_latent_dir = train_latent_dir
        self.val_latent_dir = val_latent_dir
        self.test_latent_dir = test_latent_dir
        self.text_prompt_root = text_prompt_root
        self.text_mapping_path = text_mapping_path
        self.text_prompt_subdir = text_prompt_subdir
        self.text_split = text_split
        self.text_mode = text_mode
        self.text_only_mapped = text_only_mapped
        self.text_key = text_key
        self.text_mask_key = text_mask_key
        self.tag_prompt_root = tag_prompt_root
        self.tag_prompt_subdir = tag_prompt_subdir
        self.tag_time_scale = tag_time_scale
        self.tag_sentence_style = tag_sentence_style

        self.train_transform = WaymoTargetBuilderTrain(
            train_max_num,
            enforce_map_boundary=enforce_map_boundary,
            map_boundary_margin=map_boundary_margin,
            map_boundary_gt_start_step=map_boundary_gt_start_step,
        )
        self.val_transform = WaymoTargetBuilderVal(
            enforce_map_boundary=enforce_map_boundary,
            map_boundary_margin=map_boundary_margin,
            map_boundary_gt_start_step=map_boundary_gt_start_step,
        )
        self.test_transform = WaymoTargetBuilderVal(
            enforce_map_boundary=enforce_map_boundary,
            map_boundary_margin=map_boundary_margin,
            map_boundary_gt_start_step=map_boundary_gt_start_step,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MultiDataset(
                self.train_raw_dir,
                self.train_transform,
                latent_dir=self.train_latent_dir,
                text_prompt_root=self.text_prompt_root,
                text_mapping_path=self.text_mapping_path,
                text_prompt_subdir=self.text_prompt_subdir,
                text_split=self.text_split,
                text_mode=self.text_mode,
                text_only_mapped=self.text_only_mapped,
                text_key=self.text_key,
                text_mask_key=self.text_mask_key,
                tag_prompt_root=self.tag_prompt_root,
                tag_prompt_subdir=self.tag_prompt_subdir,
                tag_time_scale=self.tag_time_scale,
                tag_sentence_style=self.tag_sentence_style,
            )
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
                latent_dir=self.val_latent_dir,
                text_prompt_root=self.text_prompt_root,
                text_mapping_path=self.text_mapping_path,
                text_prompt_subdir=self.text_prompt_subdir,
                text_split=self.text_split,
                text_mode=self.text_mode,
                text_only_mapped=self.text_only_mapped,
                text_key=self.text_key,
                text_mask_key=self.text_mask_key,
                tag_prompt_root=self.tag_prompt_root,
                tag_prompt_subdir=self.tag_prompt_subdir,
                tag_time_scale=self.tag_time_scale,
                tag_sentence_style=self.tag_sentence_style,
            )
        elif stage == "validate":
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
                latent_dir=self.val_latent_dir,
                text_prompt_root=self.text_prompt_root,
                text_mapping_path=self.text_mapping_path,
                text_prompt_subdir=self.text_prompt_subdir,
                text_split=self.text_split,
                text_mode=self.text_mode,
                text_only_mapped=self.text_only_mapped,
                text_key=self.text_key,
                text_mask_key=self.text_mask_key,
                tag_prompt_root=self.tag_prompt_root,
                tag_prompt_subdir=self.tag_prompt_subdir,
                tag_time_scale=self.tag_time_scale,
                tag_sentence_style=self.tag_sentence_style,
            )
        elif stage == "test":
            self.test_dataset = MultiDataset(
                self.test_raw_dir,
                self.test_transform,
                latent_dir=self.test_latent_dir,
                text_prompt_root=self.text_prompt_root,
                text_mapping_path=self.text_mapping_path,
                text_prompt_subdir=self.text_prompt_subdir,
                text_split=self.text_split,
                text_mode=self.text_mode,
                text_only_mapped=self.text_only_mapped,
                text_key=self.text_key,
                text_mask_key=self.text_mask_key,
                tag_prompt_root=self.tag_prompt_root,
                tag_prompt_subdir=self.tag_prompt_subdir,
                tag_time_scale=self.tag_time_scale,
                tag_sentence_style=self.tag_sentence_style,
            )
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # 0
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
