import sys
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import pickle
import glob
import numpy as np
from torch.utils.data import Dataset
from .data_utils import *
import functools

TEXT_ID_RE = re.compile(r"<([^>]+)>")
TEXT_QUOTE_RE = re.compile(r"\"([^\"]+)\"")
TAG_RE = re.compile(r"^(?P<tag>\w+)\((?P<agents>[^ ]+) at (?P<ts>\d+)-(?P<te>\d+)\)$")
AGENT_TYPE_LABELS = {
    0: "vehicle",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}
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


class WaymaxDataset(Dataset):
    """
    Dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
    """

    def __init__(
        self,
        data_dir,
        anchor_path = "data/cluster_64_center_dict.pkl",
        latent_dir = None,  # directory containing per-scenario control latents (optional)
        conditioning_mode = 'future',  # 'future' or 'latent'
        allow_missing_latents: bool = False,
        file_list: str | None = None,  # optional txt file listing pkl paths
        text_prompt_root: str | None = None,
        text_mapping_path: str | None = None,
        text_prompt_subdir: str = "70b_0.8_0.9_action_prompt_v2",
        text_split: str = "val",
        text_mode: str = "full",
        tag_prompt_root: str | None = None,
        tag_prompt_subdir: str = "auto",
        tag_time_scale: float = 0.1,
        tag_sentence_style: str = "ordered",
        attach_text_prompts: bool = False,
        filter_text_mapping: Optional[bool] = None,
        filter_text_prompts: Optional[bool] = None,
        debug_text_mapping: bool = False,
    ):
        if file_list is not None:
            with open(file_list, "r") as f:
                self.data_list = [line.strip() for line in f if line.strip()]
        else:
            self.data_list = glob.glob(data_dir+'/*') if data_dir is not None else []
        self.anchors = pickle.load(open(anchor_path, "rb"))
        self.latent_dir = latent_dir
        self.conditioning_mode = conditioning_mode
        self.allow_missing_latents = bool(allow_missing_latents)

        self._text_prompt_root = Path(text_prompt_root) if text_prompt_root else None
        self._text_mapping = None
        if text_mapping_path:
            with open(text_mapping_path, "r") as f:
                self._text_mapping = json.load(f)
        self._text_prompt_subdir = str(text_prompt_subdir)
        self._text_split = str(text_split)
        self._text_mode = str(text_mode).lower()
        self._tag_prompt_root = (
            Path(tag_prompt_root)
            if tag_prompt_root is not None
            else (Path(text_prompt_root) if text_prompt_root is not None else None)
        )
        self._tag_prompt_subdir = str(tag_prompt_subdir).lower()
        self._tag_time_scale = float(tag_time_scale)
        self._tag_sentence_style = str(tag_sentence_style).lower()
        self._text_cache: Dict[Path, List[str]] = {}
        self._tag_cache: Dict[Path, List[Dict[str, object]]] = {}
        cond_mode = str(conditioning_mode).lower()
        self._filter_text_mapping = (
            bool(filter_text_mapping)
            if filter_text_mapping is not None
            else cond_mode == "text"
        )
        self._filter_text_prompts = (
            bool(filter_text_prompts)
            if filter_text_prompts is not None
            else cond_mode == "text"
        )
        self._debug_text_mapping = bool(debug_text_mapping)
        self._attach_text_prompts_enabled = bool(
            attach_text_prompts
            or (self._text_mapping and (self._text_prompt_root or self._tag_prompt_root))
        )

        if (self._filter_text_mapping or self._filter_text_prompts) and self._text_mapping:
            self._filter_data_list_for_text()

        self.__collate_fn__ = data_collate_fn

    def _scenario_id_from_path(self, path: str) -> str:
        base = Path(path).name
        if base.startswith("scenario_") and base.endswith(".pkl"):
            return base[len("scenario_") : -len(".pkl")]
        if base.endswith(".pkl"):
            return base[:-len(".pkl")]
        return base

    def _filter_data_list_for_text(self) -> None:
        if not self.data_list:
            return
        if self._text_mapping is None:
            return

        need_mapping = bool(self._filter_text_mapping)
        need_prompts = bool(self._filter_text_prompts)
        need_text = (
            need_prompts
            and self._text_mode in ("full", "text_only")
            and self._text_prompt_root is not None
        )
        need_tag = (
            need_prompts
            and self._text_mode in ("full", "tag_only")
            and self._tag_prompt_root is not None
        )
        if not (need_mapping or need_text or need_tag):
            return

        split = self._infer_text_split()
        tag_subdir = self._infer_tag_subdir()
        text_root = None
        if need_text:
            text_root = (
                self._text_prompt_root
                / "text_prompts"
                / split
                / self._text_prompt_subdir
            )
        tag_root = None
        if need_tag:
            tag_root = self._tag_prompt_root / "tag_prompts" / tag_subdir / "tags"

        total = len(self.data_list)
        kept = 0
        missing_mapping = 0
        missing_text = 0
        missing_tag = 0
        filtered = []

        for path in self.data_list:
            scenario_id = self._scenario_id_from_path(path)
            scene_id = self._text_mapping.get(scenario_id)
            if need_mapping and scene_id is None:
                missing_mapping += 1
                continue
            if scene_id is None:
                missing_mapping += 1
                continue

            if need_text and text_root is not None:
                text_path = (
                    text_root
                    / str(self._scene_id_to_bucket(scene_id))
                    / f"{scene_id}_10_90_output.txt"
                )
                if not text_path.exists():
                    missing_text += 1
                    continue

            if need_tag and tag_root is not None:
                tag_path = (
                    tag_root
                    / str(self._scene_id_to_bucket(scene_id))
                    / f"{scene_id}.json"
                )
                if not tag_path.exists():
                    missing_tag += 1
                    continue

            filtered.append(path)
            kept += 1

        self.data_list = filtered
        if self._debug_text_mapping:
            print(
                "[WaymaxDataset] text filter: total=%d kept=%d missing_mapping=%d missing_text=%d missing_tag=%d"
                % (total, kept, missing_mapping, missing_text, missing_tag)
            )

    def __len__(self):
        return len(self.data_list)

    def _infer_text_split(self) -> str:
        if self._text_split and self._text_split != "auto":
            return self._text_split
        return "val"

    def _infer_tag_subdir(self) -> str:
        if self._tag_prompt_subdir and self._tag_prompt_subdir != "auto":
            return self._tag_prompt_subdir
        split = self._infer_text_split()
        if split == "train":
            return "waymo_train_v_action"
        return "waymo_val_v_action"

    def _scene_id_to_bucket(self, scene_id: str) -> int:
        try:
            return int(scene_id.split("_")[-1]) % 100
        except Exception:
            return 0

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

    def _parse_text_file(self, text_path: Path) -> List[str]:
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
        other_map: Dict[object, tuple[str, int]],
        agent_type_by_id: Dict[str, int],
        target_type: Optional[int],
        ego_type: Optional[int],
    ) -> str:
        target_entity = _target_entity(target_type)
        ego_entity = _ego_entity(ego_type)
        target_is_ego = ego_id is not None and int(target_id) == int(ego_id)

        counters = other_map.setdefault(
            "__counters__",
            {"vehicle": 0, "pedestrian": 0, "cyclist": 0},
        )

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

    def _attach_text_prompts(self, data, tensors):
        if self._text_mapping is None:
            return
        scenario_id = data.get("scenario_id")
        if not scenario_id:
            return

        split = self._infer_text_split()
        scene_id = self._text_mapping.get(scenario_id)

        agent_ids = data.get("agents_object_id")
        if agent_ids is None:
            agent_ids = data.get("agents_id")
        if agent_ids is None:
            return
        agent_ids_list = (
            agent_ids.tolist()
            if hasattr(agent_ids, "tolist")
            else list(agent_ids)
        )
        agent_types_data = data.get("agents_type")
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

        ego_id = data.get("ego_object_id")
        if ego_id is not None:
            if hasattr(ego_id, "item"):
                ego_id = ego_id.item()
            try:
                ego_id = int(ego_id)
            except Exception:
                ego_id = None
        ego_type = None
        if ego_id is not None:
            ego_type = agent_type_by_id.get(str(ego_id))

        prompts = [""] * len(agent_ids_list)
        prompt_mask = torch.zeros(len(agent_ids_list), dtype=torch.bool)
        if scene_id is not None:
            if self._text_mode in ("full", "text_only") and self._text_prompt_root is not None:
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
                        try:
                            agent_id_int = int(agent_id)
                        except Exception:
                            agent_id_int = agent_id
                        other_map: Dict[object, tuple[str, int]] = {}
                        rewritten: List[str] = []
                        for sentence in sentences:
                            mentions = TEXT_ID_RE.findall(sentence)
                            if not mentions:
                                continue
                            has_target = False
                            for m in mentions:
                                token = m.strip()
                                if token.lower() == "ego":
                                    if ego_id is not None and int(agent_id_int) == int(ego_id):
                                        has_target = True
                                        break
                                else:
                                    try:
                                        if int(token) == int(agent_id_int):
                                            has_target = True
                                            break
                                    except Exception:
                                        continue
                            if not has_target:
                                continue
                            target_type = agent_type_by_id.get(str(agent_id_int))
                            rewritten.append(
                                self._rewrite_sentence(
                                    sentence,
                                    int(agent_id_int) if isinstance(agent_id_int, (int, np.integer)) else 0,
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

        tensors["text_prompt"] = prompts
        tensors["text_prompt_mask"] = prompt_mask
    
    def _process(self, types):
        """
        Process the agent types and convert them into anchor vectors.

        Args:
            types (numpy.ndarray): Array of agent types.

        Returns:
            numpy.ndarray: Array of anchor vectors.
        """
        anchors = []

        for i in range(len(types)):
            if types[i] == 1:
                anchors.append(self.anchors['TYPE_VEHICLE'])
            elif types[i] == 2:
                anchors.append(self.anchors['TYPE_PEDESTRIAN'])
            elif types[i] == 3:
                anchors.append(self.anchors['TYPE_CYCLIST'])
            else:
                anchors.append(np.zeros_like(self.anchors['TYPE_VEHICLE']))

        return np.array(anchors, dtype=np.float32)
    
    def gen_tensor(self, data):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """
        
        agents_history = data['agents_history']
        agents_interested = data['agents_interested']
        agents_future = data['agents_future']
        agents_type = data['agents_type']
        agents_object_id = data.get('agents_object_id')
        ego_object_id = data.get('ego_object_id')
        traffic_light_points = data['traffic_light_points']
        polylines = data['polylines']
        polylines_valid = data['polylines_valid']
        relations = data['relations']
        anchors = self._process(agents_type)

        tensors = {
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            "anchors": torch.from_numpy(anchors)
        }
        if agents_object_id is not None:
            tensors["agents_object_id"] = torch.from_numpy(agents_object_id)
        if ego_object_id is not None:
            tensors["ego_object_id"] = torch.tensor(ego_object_id)
        
        return tensors

    def __getitem__(self, idx):
        with open(self.data_list[idx], 'rb') as f:
            data = pickle.load(f)

        tensors = self.gen_tensor(data)

        # Load corresponding control latents if in latent mode.
        # Expected file (by default): <latent_dir>/scenario_<scenario_id>_latents.pkl
        #
        # Supported formats:
        # - agent_latents: [A, Za]
        # - agent_latents_segments: [S, A, Za] (closed-loop segments)
        #
        # Notes:
        # - Scenario-Dreamer latents are retired; scenario_latents are intentionally not used.
        # - If agent_ids are present, we verify they match for safety.
        if self.conditioning_mode == 'latent' and self.latent_dir is not None:
            scenario_id = data.get('scenario_id')
            if scenario_id:
                latent_path = os.path.join(
                    self.latent_dir,
                    f'scenario_{scenario_id}_latents.pkl'
                )

                if os.path.exists(latent_path):
                    try:
                        with open(latent_path, 'rb') as f:
                            latent_data = pickle.load(f)

                        # Verify agent IDs match if provided (safety check).
                        if 'agents_id' in data and 'agent_ids' in latent_data:
                            data_ids = np.asarray(data['agents_id'])
                            latent_ids = np.asarray(latent_data['agent_ids'])
                            if latent_ids.shape[0] > data_ids.shape[0]:
                                raise ValueError(
                                    f"Latent agent_ids longer than data agents_id for scenario {scenario_id}"
                                )
                            if not np.array_equal(data_ids[: latent_ids.shape[0]], latent_ids):
                                raise ValueError(
                                    f"Latent agent_ids mismatch for scenario {scenario_id}"
                                )

                        if 'agent_latents_segments' in latent_data:
                            tensors['agent_latents_segments'] = torch.from_numpy(latent_data['agent_latents_segments']).float()
                            tensors['num_segments'] = int(latent_data.get('num_segments', tensors['agent_latents_segments'].shape[0]))
                            tensors['agent_latents'] = tensors['agent_latents_segments'][0]
                        elif 'agent_latents' in latent_data:
                            tensors['agent_latents'] = torch.from_numpy(latent_data['agent_latents']).float()
                        else:
                            raise KeyError("Latent file must contain 'agent_latents' or 'agent_latents_segments'")
                    except Exception as e:
                        if self.allow_missing_latents:
                            print(f"Warning: Failed to load latents for scenario {scenario_id}: {e}")
                        else:
                            raise
                else:
                    if self.allow_missing_latents:
                        print(f"Warning: Latent file not found for scenario {scenario_id}: {latent_path}")
                    else:
                        raise FileNotFoundError(f"Latent file not found for scenario {scenario_id}: {latent_path}")

        if self._attach_text_prompts_enabled:
            self._attach_text_prompts(data, tensors)

        return tensors


class WaymaxTestDataset(WaymaxDataset):
    """
    Test dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
        max_object (int, optional): Maximum number of objects. Defaults to 16.
        max_polylines (int, optional): Maximum number of polylines. Defaults to 256.
        history_length (int, optional): Length of history. Defaults to 11.
        num_points_polyline (int, optional): Number of points in each polyline. Defaults to 30.
    """

    def __init__(
        self,
        data_dir: str,
        anchor_path = "data/cluster_64_center_dict.pkl",
        max_object: int = 16,
        max_map_points: int = 3000,
        max_polylines: int = 256,
        history_length: int = 11,
        num_points_polyline: int = 30,
    ) -> None:
        super().__init__(data_dir, anchor_path)

        self.max_object = max_object
        self.max_polylines = max_polylines
        self.max_map_points = max_map_points
        self.history_length = history_length
        self.num_points_polyline = num_points_polyline
        
        self.base_path = os.path.dirname(os.path.abspath(self.data_list[0])) if len(self.data_list) > 0 else None
                
    def process_scenario(self, scenario_raw, current_index: int = 10,
                        use_log: bool = True, selected_agents=None,
                        remove_history=False):
        """
        Process a scenario and generate tensors.

        Args:
            scenario_raw (dict): Raw scenario data.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.
            selected_agents (list, optional): List of selected agents. Defaults to None.

        Returns:
            dict: Dictionary of tensors.
        """
        data_dict = data_process_scenario(
            scenario_raw,
            current_index=current_index,
            max_num_objects=self.max_object,
            max_polylines=self.max_polylines,
            num_points_polyline=self.num_points_polyline,
            use_log=use_log,
            selected_agents=selected_agents,
            remove_history=remove_history,
        )
        
        data_dict['anchors'] = self._process(data_dict['agents_type'])

        return data_dict
        
    def reset_agent_length(self,max_object):
        """
        Reset the maximum number of objects.

        Args:
            max_object (int): Maximum number of objects.
        """
        self.max_object = max_object
        
    def get_scenario_by_id(
        self, scenario_id,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False
    ):
        """
        Get a scenario by its ID.

        Args:
            scenario_id (int): Scenario ID.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        file_path = os.path.join(self.base_path, f"scenario_{scenario_id}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if 'scenario_raw' in data:
            scenario_raw = data['scenario_raw']
        elif 'scenario' in data:
            scenario_raw = data['scenario']
        else:
            raise ValueError("scenario_raw not found")
        
        data_dict = self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )
        
        return scenario_id, scenario_raw, data_dict
    
    def get_scenario_by_index(
        self, index,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False
    ):
        """
        Get a scenario by its index.

        Args:
            index (int): Scenario index.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        filename = self.data_list[index]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if 'scenario_raw' in data:
            scenario_raw = data['scenario_raw']
            scenario_id = data['scenario_id']
        elif 'scenario' in data:
            scenario_raw = data['scenario']
            scenario_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
        else:
            raise ValueError("scenario_raw not found")
        
        
        data_dict = self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )
        
        return scenario_id, scenario_raw, data_dict
    
    def __getitem__(self, idx):
        _, _, data_dict = self.get_scenario_by_index(idx)
        return data_dict       
    
    
