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

import json
import pickle
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

TEXT_ID_RE = re.compile(r"<([^>]+)>")
TEXT_QUOTE_RE = re.compile(r"\"([^\"]+)\"")
TAG_RE = re.compile(r"^(?P<tag>\w+)\((?P<agents>[^ ]+) at (?P<ts>\d+)-(?P<te>\d+)\)$")
AGENT_TYPE_LABELS = {
    0: "vehicle",
    1: "pedestrian",
    2: "cyclist",
}


def _agent_type_label(agent_type: Optional[int]) -> str:
    if agent_type is None:
        return AGENT_TYPE_LABELS[0]
    try:
        return AGENT_TYPE_LABELS[int(agent_type)]
    except (ValueError, KeyError, TypeError):
        return AGENT_TYPE_LABELS[0]


def _target_entity(agent_type: Optional[int]) -> str:
    return f"the target {_agent_type_label(agent_type)}"


def _ego_entity(agent_type: Optional[int]) -> str:
    return f"the ego {_agent_type_label(agent_type)}"
TAG_PHRASES = {
    "Parked": "parked",
    "Straight": "moving straight",
    "LeftTurn": "turning left",
    "RightTurn": "turning right",
    "LeftLaneChange": "changing lanes left",
    "RightLaneChange": "changing lanes right",
    "Accelerate": "accelerating",
    "Decelerate": "decelerating",
    "Stopping": "slowing down to a stop",
    "KeepSpeed": "keeping a steady speed",
}
TAG_INTEGRATE_TOLERANCE = 5
TAG_MIN_DURATION = 10
TAG_EXCLUSION_GROUPS = {
    "Parked": [
        "Straight",
        "LeftTurn",
        "RightTurn",
        "LeftLaneChange",
        "RightLaneChange",
        "Accelerate",
        "Decelerate",
        "Stopping",
        "KeepSpeed",
    ],
    "Straight": ["LeftTurn", "RightTurn", "Parked", "LeftLaneChange", "RightLaneChange"],
    "LeftTurn": ["RightTurn", "Straight", "LeftLaneChange", "RightLaneChange"],
    "RightTurn": ["LeftTurn", "Straight", "LeftLaneChange", "RightLaneChange"],
    "LeftLaneChange": ["RightLaneChange", "Straight", "LeftTurn", "RightTurn", "Parked"],
    "RightLaneChange": ["LeftLaneChange", "Straight", "LeftTurn", "RightTurn", "Parked"],
    "Accelerate": ["Parked"],
    "Decelerate": ["Parked"],
    "Stopping": ["Parked"],
    "KeepSpeed": ["Parked"],
}
TAG_PRIORITY = {
    "LeftTurn": 1,
    "RightTurn": 1,
    "LeftLaneChange": 1,
    "RightLaneChange": 1,
    "Straight": 2,
    "Accelerate": 3,
    "Decelerate": 3,
    "Stopping": 3,
    "KeepSpeed": 3,
    "Parked": 4,
}


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        latent_dir: Optional[str] = None,
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
    ) -> None:
        raw_dir = Path(raw_dir)
        self._raw_dir = raw_dir
        self._raw_paths = [p.as_posix() for p in sorted(raw_dir.glob("*"))]
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        self._latent_dir = Path(latent_dir) if latent_dir is not None else None
        self._text_prompt_root = (
            Path(text_prompt_root) if text_prompt_root is not None else None
        )
        self._tag_prompt_root = (
            Path(tag_prompt_root)
            if tag_prompt_root is not None
            else (Path(text_prompt_root) if text_prompt_root is not None else None)
        )
        self._text_mapping = None
        if text_mapping_path is not None:
            with Path(text_mapping_path).open("r") as f:
                self._text_mapping = json.load(f)
        if text_only_mapped:
            if self._text_mapping is None:
                log.warning(
                    "text_only_mapped=True but text_mapping_path is None; keeping full dataset."
                )
            else:
                before = len(self._raw_paths)
                mapped_ids = set(self._text_mapping.keys())
                filtered = []
                for p in self._raw_paths:
                    sid = Path(p).stem
                    if sid in mapped_ids:
                        filtered.append(p)
                self._raw_paths = filtered
                self._num_samples = len(self._raw_paths)
                log.info(
                    f"Filtered dataset to mapped scenarios: {before} -> {self._num_samples}"
                )
        self._text_key = str(text_key)
        self._text_mask_key = str(text_mask_key)
        self._text_split = str(text_split).lower()
        self._text_prompt_subdir = str(text_prompt_subdir)
        self._text_mode = str(text_mode).lower()
        self._text_cache = {}
        self._tag_prompt_subdir = str(tag_prompt_subdir).lower()
        self._tag_time_scale = float(tag_time_scale)
        self._tag_sentence_style = str(tag_sentence_style).lower()
        self._tag_cache = {}

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        self._log_text_mapping_stats()
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        if self._latent_dir is not None:
            data = self._attach_latents(data)

        if self._text_prompt_root is not None and self._text_mapping is not None:
            data = self._attach_text_prompts(data)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        return data

    def _attach_latents(self, data):
        scenario_id = data.get("scenario_id")
        if not scenario_id:
            raise ValueError("scenario_id missing; cannot attach latents")

        latent_path = self._latent_dir / f"scenario_{scenario_id}_latents.pkl"
        if not latent_path.exists():
            raise FileNotFoundError(f"latent file not found: {latent_path}")

        with open(latent_path, "rb") as handle:
            payload = pickle.load(handle)

        agent_latents = payload.get("agent_latents")
        if agent_latents is None:
            raise ValueError(f"agent_latents missing in {latent_path}")

        agent_latents = torch.as_tensor(agent_latents, dtype=torch.float32)
        if agent_latents.dim() != 3:
            raise ValueError(
                f"agent_latents must be [S,A,Z], got {agent_latents.shape} in {latent_path}"
            )

        num_steps, n_latent_agents, latent_dim = agent_latents.shape
        agent_latents = agent_latents.permute(1, 0, 2)  # [A, S, Z]

        agent_ids_data = data["agent"]["id"]
        agent_ids_list = (
            agent_ids_data.tolist()
            if hasattr(agent_ids_data, "tolist")
            else list(agent_ids_data)
        )
        n_agent = len(agent_ids_list)

        agent_ids_latent = payload.get("agent_ids")
        if agent_ids_latent is None:
            if n_latent_agents != n_agent:
                raise ValueError(
                    f"latent agent_ids missing and count mismatch ({n_latent_agents} vs {n_agent}) "
                    f"for {latent_path}"
                )
            data["agent"]["latent"] = agent_latents
            data["agent"]["latent_mask"] = torch.ones(n_agent, dtype=torch.bool)
        else:
            agent_ids_latent = (
                agent_ids_latent.tolist()
                if hasattr(agent_ids_latent, "tolist")
                else list(agent_ids_latent)
            )
            latent_id_to_idx = {int(aid): idx for idx, aid in enumerate(agent_ids_latent)}
            latent_out = torch.zeros(
                (n_agent, num_steps, latent_dim), dtype=torch.float32
            )
            latent_mask = torch.zeros(n_agent, dtype=torch.bool)
            for i, agent_id in enumerate(agent_ids_list):
                idx = latent_id_to_idx.get(int(agent_id))
                if idx is None:
                    continue
                latent_out[i] = agent_latents[idx]
                latent_mask[i] = True
            data["agent"]["latent"] = latent_out
            data["agent"]["latent_mask"] = latent_mask

        scene_latents = payload.get("scene_latents")
        if scene_latents is not None:
            data["latent_scene"] = torch.as_tensor(
                scene_latents, dtype=torch.float32
            )

        return data

    def _infer_text_split(self) -> str:
        if self._text_split and self._text_split != "auto":
            return self._text_split
        raw_name = self._raw_dir.as_posix().lower()
        if "train" in raw_name:
            return "train"
        if "val" in raw_name:
            return "val"
        return "val"

    def _log_text_mapping_stats(self, sample_size: int = 200) -> None:
        if self._text_mapping is None or self._text_prompt_root is None:
            return
        total = len(self._raw_paths)
        if total == 0:
            return
        mapped_ids = set(self._text_mapping.keys())
        mapped = sum(1 for p in self._raw_paths if Path(p).stem in mapped_ids)
        split = self._infer_text_split()
        log.info(
            f"Text mapping coverage ({split}): {mapped}/{total} ({mapped/total:.2%})"
        )

        # Sample for agent-id alignment coverage
        mapped_paths = [p for p in self._raw_paths if Path(p).stem in mapped_ids]
        if not mapped_paths:
            return
        sample_n = min(sample_size, len(mapped_paths))
        sample_paths = random.sample(mapped_paths, sample_n)

        text_root = (
            self._text_prompt_root / "text_prompts" / split / self._text_prompt_subdir
        )
        total_ids = 0
        hit_ids = 0
        scenarios_missing_any = 0
        missing_text_files = 0

        def _scene_bucket(scene_id: str) -> int:
            try:
                return int(scene_id.split("_")[-1]) % 100
            except Exception:
                return 0

        for p in sample_paths:
            sid = Path(p).stem
            scene_id = self._text_mapping.get(sid)
            if scene_id is None:
                continue
            text_path = (
                text_root
                / str(_scene_bucket(scene_id))
                / f"{scene_id}_10_90_output.txt"
            )
            if not text_path.exists():
                missing_text_files += 1
                continue
            sentences = self._parse_text_file(text_path)
            if not sentences:
                continue
            with open(p, "rb") as handle:
                data = pickle.load(handle)
            agent_ids = data.get("agent", {}).get("id", [])
            agent_ids = (
                agent_ids.tolist() if hasattr(agent_ids, "tolist") else list(agent_ids)
            )
            agent_id_set = set(int(x) for x in agent_ids)
            missing = False
            for sentence in sentences:
                for token in TEXT_ID_RE.findall(sentence):
                    token = token.strip()
                    if token.lower() == "ego":
                        continue
                    try:
                        token_id = int(token)
                    except ValueError:
                        continue
                    total_ids += 1
                    if token_id in agent_id_set:
                        hit_ids += 1
                    else:
                        missing = True
            if missing:
                scenarios_missing_any += 1

        hit_rate = (hit_ids / total_ids) if total_ids else 0.0
        log.info(
            "Text agent-id coverage sample "
            f"({split}, n={sample_n}): hit={hit_rate:.2%} "
            f"({hit_ids}/{total_ids}), "
            f"missing_scenarios={scenarios_missing_any}/{sample_n}, "
            f"missing_text_files={missing_text_files}/{sample_n}"
        )

    def _infer_tag_subdir(self) -> str:
        if self._tag_prompt_subdir and self._tag_prompt_subdir != "auto":
            return self._tag_prompt_subdir
        split = self._infer_text_split()
        if split == "train":
            return "waymo_train_v_action"
        return "waymo_val_v_action"

    def _scene_id_to_bucket(self, scene_id: str) -> int:
        return int(scene_id.split("_")[-1]) % 100

    def _parse_text_file(self, text_path: Path):
        cached = self._text_cache.get(text_path)
        if cached is not None:
            return cached
        text = text_path.read_text()
        sentences = TEXT_QUOTE_RE.findall(text)
        if not sentences:
            sentences = [line.strip() for line in text.splitlines() if line.strip()]
        sentences = [self._normalize_text_seconds(s) for s in sentences]
        sentences = [self._strip_sentence_prefix(s) for s in sentences]
        if len(self._text_cache) > 2048:
            self._text_cache.clear()
        self._text_cache[text_path] = sentences
        return sentences

    def _strip_sentence_prefix(self, sentence: str) -> str:
        return re.sub(r"^\s*\d+\.\s*", "", sentence)

    def _normalize_text_seconds(self, sentence: str) -> str:
        def _replace_space(match: re.Match) -> str:
            value = int(match.group(1)) / 10.0
            return f"{value:.1f} seconds"

        def _replace_hyphen(match: re.Match) -> str:
            value = int(match.group(1)) / 10.0
            return f"{value:.1f}-second"

        sentence = re.sub(r"\b(\d{1,3})\s*seconds?\b", _replace_space, sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"\b(\d{1,3})-second\b", _replace_hyphen, sentence, flags=re.IGNORECASE)
        return sentence

    def _parse_tag_file(self, tag_path: Path) -> List[Dict[str, object]]:
        cached = self._tag_cache.get(tag_path)
        if cached is not None:
            return cached
        with tag_path.open("r") as f:
            tags = json.load(f)
        results = []
        for entry in tags:
            m = TAG_RE.match(entry)
            if not m:
                continue
            tag = m.group("tag")
            if tag.endswith("Temporal"):
                tag = tag[: -len("Temporal")]
            agents = [a.strip() for a in m.group("agents").split(",")]
            ts = int(m.group("ts"))
            te = int(m.group("te"))
            for agent in agents:
                if agent:
                    results.append(
                        {
                            "agent": agent,
                            "action": tag,
                            "ts": ts,
                            "te": te,
                        }
                    )
        if len(self._tag_cache) > 2048:
            self._tag_cache.clear()
        self._tag_cache[tag_path] = results
        return results

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= cur_end + 1e-6:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged

    def _integrate_tags(self, tags: List[Dict[str, object]], tolerance: int) -> List[Dict[str, object]]:
        per_key: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for tag in tags:
            agent = str(tag["agent"])
            action = str(tag["action"])
            ts = int(tag["ts"])
            te = int(tag["te"])
            key = (agent, action)
            per_key.setdefault(key, []).append((ts, te))

        merged_tags: List[Dict[str, object]] = []
        for (agent, action), intervals in per_key.items():
            intervals = sorted(intervals, key=lambda x: x[0])
            cur_start, cur_end = intervals[0]
            for start, end in intervals[1:]:
                if cur_end + tolerance >= start:
                    cur_end = max(cur_end, end)
                else:
                    merged_tags.append(
                        {"agent": agent, "action": action, "ts": cur_start, "te": cur_end}
                    )
                    cur_start, cur_end = start, end
            merged_tags.append(
                {"agent": agent, "action": action, "ts": cur_start, "te": cur_end}
            )
        return merged_tags

    def _remove_short_tags(
        self, tags: List[Dict[str, object]], min_duration: int
    ) -> List[Dict[str, object]]:
        return [
            tag
            for tag in tags
            if int(tag["te"]) - int(tag["ts"]) >= int(min_duration)
        ]

    def _resolve_tag_conflicts(self, tags: List[Dict[str, object]]) -> List[Dict[str, object]]:
        per_agent: Dict[str, List[Dict[str, object]]] = {}
        for tag in tags:
            per_agent.setdefault(str(tag["agent"]), []).append(tag)

        resolved: List[Dict[str, object]] = []
        for agent, agent_tags in per_agent.items():
            sorted_tags = sorted(agent_tags, key=lambda x: int(x["ts"]))
            current: List[Dict[str, object]] = []
            for tag in sorted_tags:
                new_start = int(tag["ts"])
                new_end = int(tag["te"])
                new_action = str(tag["action"])
                new_priority = TAG_PRIORITY.get(new_action, float("inf"))
                adjusted: List[Dict[str, object]] = []
                for cur in current:
                    cur_start = int(cur["ts"])
                    cur_end = int(cur["te"])
                    cur_action = str(cur["action"])
                    cur_priority = TAG_PRIORITY.get(cur_action, float("inf"))
                    overlap = max(cur_start, new_start) < min(cur_end, new_end)
                    if (
                        overlap
                        and new_action in TAG_EXCLUSION_GROUPS.get(cur_action, [])
                    ):
                        if cur_priority < new_priority:
                            new_start = cur_end
                        elif new_priority < cur_priority:
                            if cur_start < new_start:
                                adjusted.append(
                                    {
                                        "agent": agent,
                                        "action": cur_action,
                                        "ts": cur_start,
                                        "te": new_start,
                                    }
                                )
                            cur_end = new_start
                        else:
                            if new_start > cur_start:
                                adjusted.append(
                                    {
                                        "agent": agent,
                                        "action": cur_action,
                                        "ts": cur_start,
                                        "te": new_start,
                                    }
                                )
                                cur_end = new_start
                    if cur_start < cur_end:
                        adjusted.append(
                            {
                                "agent": agent,
                                "action": cur_action,
                                "ts": cur_start,
                                "te": cur_end,
                            }
                        )
                current = adjusted
                if new_start < new_end:
                    current.append(
                        {
                            "agent": agent,
                            "action": new_action,
                            "ts": new_start,
                            "te": new_end,
                        }
                    )

            if current:
                current.sort(key=lambda x: int(x["ts"]))
                merged = [current[0]]
                for tag in current[1:]:
                    last = merged[-1]
                    if (
                        tag["action"] == last["action"]
                        and int(tag["ts"]) <= int(last["te"])
                    ):
                        last["te"] = max(int(last["te"]), int(tag["te"]))
                    else:
                        merged.append(tag)
                resolved.extend(merged)
        return resolved

    def _drop_parked_when_multiple(self, tags: List[Dict[str, object]]) -> List[Dict[str, object]]:
        per_agent: Dict[str, List[Dict[str, object]]] = {}
        for tag in tags:
            per_agent.setdefault(str(tag["agent"]), []).append(tag)

        kept: List[Dict[str, object]] = []
        for agent, agent_tags in per_agent.items():
            if len(agent_tags) <= 1:
                kept.extend(agent_tags)
                continue
            for tag in agent_tags:
                if str(tag["action"]) == "Parked":
                    continue
                kept.append(tag)
        return kept

    def _process_tags(self, tags: List[Dict[str, object]]) -> List[Dict[str, object]]:
        filtered = [t for t in tags if str(t["action"]) in TAG_PHRASES]
        if not filtered:
            return []
        integrated = self._integrate_tags(filtered, TAG_INTEGRATE_TOLERANCE)
        long_tags = self._remove_short_tags(integrated, TAG_MIN_DURATION)
        if not long_tags:
            return []
        resolved = self._resolve_tag_conflicts(long_tags)
        if not resolved:
            return []
        return self._drop_parked_when_multiple(resolved)

    def _build_tag_sentences(
        self,
        scene_id: str,
        agent_ids_list: List[object],
        ego_id: Optional[object],
        agent_type_by_id: Dict[str, int],
    ) -> Dict[str, str]:
        if self._tag_prompt_root is None:
            return {}
        tag_subdir = self._infer_tag_subdir()
        tag_root = self._tag_prompt_root / "tag_prompts" / tag_subdir / "tags"
        tag_path = tag_root / str(self._scene_id_to_bucket(scene_id)) / f"{scene_id}.json"
        if not tag_path.exists():
            return {}

        tags = self._parse_tag_file(tag_path)
        if not tags:
            return {}
        tags = self._process_tags(tags)
        if not tags:
            return {}

        agent_set = {str(a) for a in agent_ids_list}
        if ego_id is not None:
            agent_set.add(str(ego_id))

        per_agent: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        for tag in tags:
            agent = str(tag["agent"])
            if agent == "ego":
                if ego_id is None:
                    continue
                agent = str(ego_id)
            if agent not in agent_set:
                continue
            action = str(tag["action"])
            phrase = TAG_PHRASES.get(action)
            if phrase is None:
                continue
            ts = float(tag["ts"]) * self._tag_time_scale
            te = float(tag["te"]) * self._tag_time_scale
            if te < ts:
                ts, te = te, ts
            per_agent.setdefault(agent, {}).setdefault(action, []).append((ts, te))

        out: Dict[str, str] = {}
        for agent, action_map in per_agent.items():
            target_entity = _target_entity(agent_type_by_id.get(agent))
            if self._tag_sentence_style == "ordered":
                start_groups: Dict[float, List[str]] = {}
                for action, intervals in action_map.items():
                    for ts, _ in intervals:
                        start_groups.setdefault(ts, []).append(action)
                if not start_groups:
                    continue
                ordered_starts = sorted(start_groups.keys())
                steps: List[str] = []
                for ts in ordered_starts:
                    actions = start_groups[ts]
                    # Deduplicate and order by priority for stability.
                    uniq = sorted(
                        set(actions),
                        key=lambda a: (TAG_PRIORITY.get(a, 99), a),
                    )
                    phrases = [TAG_PHRASES[a] for a in uniq]
                    if not phrases:
                        continue
                    if len(phrases) == 1:
                        action_text = phrases[0]
                    elif len(phrases) == 2:
                        action_text = f"{phrases[0]} and {phrases[1]}"
                    else:
                        action_text = f"{', '.join(phrases[:-1])}, and {phrases[-1]}"
                    steps.append(f"{target_entity} is {action_text}.")

                if not steps:
                    continue
                if len(steps) == 1:
                    out[agent] = steps[0]
                    continue
                ordered_steps: List[str] = []
                for idx, step in enumerate(steps):
                    if idx == 0:
                        ordered_steps.append(step)
                    else:
                        ordered_steps.append(f"Then, {step}")
                out[agent] = " ".join(ordered_steps)
                continue

            segments = []
            for action, intervals in action_map.items():
                merged = self._merge_intervals(intervals)
                phrase = TAG_PHRASES[action]
                for ts, te in merged:
                    segments.append(
                        (ts, f"{phrase} between {ts:.1f} seconds and {te:.1f} seconds")
                    )
            if not segments:
                continue
            segments.sort(key=lambda x: x[0])
            phrases = [seg for _, seg in segments]
            if len(phrases) == 1:
                sentence = f"{target_entity} is {phrases[0]}."
            elif len(phrases) == 2:
                sentence = f"{target_entity} is {phrases[0]} and {phrases[1]}."
            else:
                sentence = f"{target_entity} is {', '.join(phrases[:-1])}, and {phrases[-1]}."
            out[agent] = sentence
        return out

    def _rewrite_sentence(
        self,
        sentence: str,
        target_id: int,
        ego_id: Optional[int],
        other_map: dict,
        agent_type_by_id: Dict[str, int],
        target_type: Optional[int],
        ego_type: Optional[int],
    ) -> str:
        target_is_ego = ego_id is not None and int(target_id) == int(ego_id)
        target_entity = _target_entity(target_type)
        ego_entity = _ego_entity(ego_type)
        counters = other_map.setdefault("__counters__", {})

        def _replace(match: re.Match) -> str:
            raw = match.group(1).strip()
            if raw.lower() == "ego":
                if target_is_ego:
                    return target_entity
                key = "ego"
            else:
                try:
                    key = int(raw)
                except ValueError:
                    key = raw
            if isinstance(key, int) and key == int(target_id):
                return target_entity
            if key == "ego":
                return ego_entity
            if key not in other_map:
                other_type = agent_type_by_id.get(str(key))
                other_label = _agent_type_label(other_type)
                counters[other_label] = counters.get(other_label, 0) + 1
                other_map[key] = (other_label, counters[other_label])
            other_label, other_idx = other_map[key]
            return f"{other_label} {other_idx}"

        return TEXT_ID_RE.sub(_replace, sentence)

    def _attach_text_prompts(self, data):
        if self._text_key in data.get("agent", {}):
            return data
        scenario_id = data.get("scenario_id")
        if not scenario_id:
            return data

        split = self._infer_text_split()
        agent_ids = data["agent"]["id"]
        agent_ids_list = (
            agent_ids.tolist()
            if hasattr(agent_ids, "tolist")
            else list(agent_ids)
        )
        agent_types_data = data["agent"].get("type")
        if agent_types_data is not None:
            agent_types_list = (
                agent_types_data.tolist()
                if hasattr(agent_types_data, "tolist")
                else list(agent_types_data)
            )
        else:
            agent_types_list = [0] * len(agent_ids_list)
        agent_type_by_id: Dict[str, int] = {
            str(aid): int(at)
            for aid, at in zip(agent_ids_list, agent_types_list)
        }
        role = data["agent"]["role"]
        if hasattr(role, "numpy"):
            role = role.numpy()
        ego_candidates = [i for i, v in enumerate(role) if v[0]]
        ego_id = agent_ids_list[ego_candidates[0]] if ego_candidates else None
        ego_type = (
            int(agent_types_list[ego_candidates[0]])
            if ego_candidates and ego_candidates[0] < len(agent_types_list)
            else None
        )

        scene_id = self._text_mapping.get(scenario_id) if self._text_mapping else None
        prompts = [""] * len(agent_ids_list)
        prompt_mask = torch.zeros(len(agent_ids_list), dtype=torch.bool)
        if scene_id is not None:
            if self._text_mode in ("full", "text_only"):
                text_root = (
                    self._text_prompt_root
                    / "text_prompts"
                    / split
                    / self._text_prompt_subdir
                )
                text_path = (
                    text_root
                    / str(self._scene_id_to_bucket(scene_id))
                    / f"{scene_id}_10_90_output.txt"
                )
                if text_path.exists():
                    sentences = self._parse_text_file(text_path)
                    for idx, agent_id in enumerate(agent_ids_list):
                        other_map = {}
                        rewritten = []
                        for sentence in sentences:
                            mentions = TEXT_ID_RE.findall(sentence)
                            if not mentions:
                                continue
                            has_target = False
                            for m in mentions:
                                token = m.strip()
                                if token.lower() == "ego":
                                    if ego_id is not None and int(agent_id) == int(ego_id):
                                        has_target = True
                                        break
                                else:
                                    try:
                                        if int(token) == int(agent_id):
                                            has_target = True
                                            break
                                    except ValueError:
                                        continue
                            if not has_target:
                                continue
                            target_type = agent_type_by_id.get(str(agent_id))
                            rewritten.append(
                                self._rewrite_sentence(
                                    sentence,
                                    int(agent_id),
                                    ego_id,
                                    other_map,
                                    agent_type_by_id,
                                    target_type,
                                    ego_type,
                                )
                            )
                        if rewritten:
                            prompts[idx] = " ".join(rewritten)
                            prompt_mask[idx] = True

            if self._text_mode in ("full", "tag_only"):
                tag_sentences = self._build_tag_sentences(
                    scene_id,
                    agent_ids_list,
                    ego_id,
                    agent_type_by_id,
                )
                if tag_sentences:
                    for idx, agent_id in enumerate(agent_ids_list):
                        tag_sentence = tag_sentences.get(str(agent_id))
                        if not tag_sentence:
                            continue
                        if prompts[idx]:
                            prompts[idx] = f"{prompts[idx]} {tag_sentence}"
                        else:
                            prompts[idx] = tag_sentence
                        prompt_mask[idx] = True

        data["agent"][self._text_key] = prompts
        data["agent"][self._text_mask_key] = prompt_mask
        return data
