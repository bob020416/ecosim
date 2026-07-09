import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

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


def _strip_sentence_prefix(sentence: str) -> str:
    return re.sub(r"^\s*\d+\.\s*", "", sentence)


def _normalize_text_seconds(sentence: str) -> str:
    def _replace_space(match: re.Match) -> str:
        value = int(match.group(1)) / 10.0
        return f"{value:.1f} seconds"

    def _replace_hyphen(match: re.Match) -> str:
        value = int(match.group(1)) / 10.0
        return f"{value:.1f}-second"

    sentence = re.sub(r"\b(\d{1,3})\s*seconds?\b", _replace_space, sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\b(\d{1,3})-second\b", _replace_hyphen, sentence, flags=re.IGNORECASE)
    return sentence


def _parse_text_file(text_path: Path) -> List[str]:
    text = text_path.read_text()
    sentences = TEXT_QUOTE_RE.findall(text)
    if not sentences:
        sentences = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = [_normalize_text_seconds(s) for s in sentences]
    sentences = [_strip_sentence_prefix(s) for s in sentences]
    return sentences


def _parse_custom_text(
    custom_text: Optional[str],
    custom_text_file: Optional[Path],
    delimiter: str,
) -> List[str]:
    if custom_text_file is not None:
        if not custom_text_file.exists():
            raise RuntimeError(f"Custom text file not found: {custom_text_file}")
        lines = [line.strip() for line in custom_text_file.read_text().splitlines()]
        sentences = [line for line in lines if line]
    else:
        text = (custom_text or "").strip()
        if not text:
            return []
        if delimiter and delimiter in text:
            sentences = [seg.strip() for seg in text.split(delimiter) if seg.strip()]
        else:
            sentences = [text]
    sentences = [_strip_sentence_prefix(s) for s in sentences]
    return sentences


def _rewrite_sentence(
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


def _parse_tag_file(tag_path: Path) -> List[Dict[str, object]]:
    tags = json.loads(tag_path.read_text())
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
                results.append({"agent": agent, "action": tag, "ts": ts, "te": te})
    return results


def _infer_tag_subdir(text_split: str, tag_prompt_subdir: str) -> str:
    tag_subdir = str(tag_prompt_subdir).lower()
    if tag_subdir and tag_subdir != "auto":
        return tag_subdir
    split = str(text_split).lower()
    if split == "train":
        return "waymo_train_v_action"
    return "waymo_val_v_action"


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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


def _integrate_tags(tags: List[Dict[str, object]], tolerance: int) -> List[Dict[str, object]]:
    per_key: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for tag in tags:
        agent = str(tag["agent"])
        action = str(tag["action"])
        ts = int(tag["ts"])
        te = int(tag["te"])
        per_key.setdefault((agent, action), []).append((ts, te))

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


def _remove_short_tags(tags: List[Dict[str, object]], min_duration: int) -> List[Dict[str, object]]:
    return [
        tag
        for tag in tags
        if int(tag["te"]) - int(tag["ts"]) >= int(min_duration)
    ]


def _resolve_tag_conflicts(tags: List[Dict[str, object]]) -> List[Dict[str, object]]:
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
                if overlap and new_action in TAG_EXCLUSION_GROUPS.get(cur_action, []):
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
                if tag["action"] == last["action"] and int(tag["ts"]) <= int(last["te"]):
                    last["te"] = max(int(last["te"]), int(tag["te"]))
                else:
                    merged.append(tag)
            resolved.extend(merged)
    return resolved


def _drop_parked_when_multiple(tags: List[Dict[str, object]]) -> List[Dict[str, object]]:
    per_agent: Dict[str, List[Dict[str, object]]] = {}
    for tag in tags:
        per_agent.setdefault(str(tag["agent"]), []).append(tag)

    kept: List[Dict[str, object]] = []
    for _, agent_tags in per_agent.items():
        if len(agent_tags) <= 1:
            kept.extend(agent_tags)
            continue
        for tag in agent_tags:
            if str(tag["action"]) == "Parked":
                continue
            kept.append(tag)
    return kept


def _process_tags(tags: List[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered = [t for t in tags if str(t["action"]) in TAG_PHRASES]
    if not filtered:
        return []
    integrated = _integrate_tags(filtered, TAG_INTEGRATE_TOLERANCE)
    long_tags = _remove_short_tags(integrated, TAG_MIN_DURATION)
    if not long_tags:
        return []
    resolved = _resolve_tag_conflicts(long_tags)
    if not resolved:
        return []
    return _drop_parked_when_multiple(resolved)


def _build_tag_sentences(
    tags: List[Dict[str, object]],
    agent_ids: List[int],
    ego_id: Optional[int],
    agent_type_by_id: Dict[str, int],
    *,
    time_scale: float = 0.1,
    tag_sentence_style: str = "ordered",
) -> Dict[int, str]:
    tags = _process_tags(tags)
    if not tags:
        return {}

    per_agent: Dict[int, Dict[str, List[Tuple[float, float]]]] = {}
    for tag in tags:
        agent = str(tag["agent"])
        if agent == "ego":
            if ego_id is None:
                continue
            agent = str(ego_id)
        try:
            agent_id = int(agent)
        except ValueError:
            continue
        if agent_id not in agent_ids:
            continue
        action = str(tag["action"])
        phrase = TAG_PHRASES.get(action)
        if phrase is None:
            continue
        ts = float(tag["ts"]) * time_scale
        te = float(tag["te"]) * time_scale
        if te < ts:
            ts, te = te, ts
        per_agent.setdefault(agent_id, {}).setdefault(action, []).append((ts, te))

    out: Dict[int, str] = {}
    for agent_id, action_map in per_agent.items():
        target_entity = _target_entity(agent_type_by_id.get(str(agent_id)))
        if str(tag_sentence_style).lower() == "ordered":
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
                phrases = [TAG_PHRASES[a] for a in uniq if a in TAG_PHRASES]
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
                out[agent_id] = steps[0]
                continue

            ordered_steps: List[str] = []
            for idx, step in enumerate(steps):
                if idx == 0:
                    ordered_steps.append(step)
                else:
                    ordered_steps.append(f"Then, {step}")
            out[agent_id] = " ".join(ordered_steps)
            continue

        segments = []
        for action, intervals in action_map.items():
            merged = _merge_intervals(intervals)
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
        out[agent_id] = sentence
    return out


def _load_text_mapping(mapping_path: Path) -> Dict[str, str]:
    if not mapping_path.exists():
        raise RuntimeError(f"Text mapping not found: {mapping_path}")
    return json.loads(mapping_path.read_text())


def _build_control_text_for_batch(
    batch: HeteroData,
    scenario_id: str,
    *,
    text_prompt_root: Path,
    text_prompt_subdir: str,
    text_split: str,
    tag_prompt_subdir: str,
    tag_time_scale: float,
    text_mode: str,
    tag_sentence_style: str,
    text_mapping: Dict[str, str],
) -> Tuple[List[str], torch.Tensor]:
    scene_id = text_mapping.get(scenario_id)
    agent_ids = batch["agent"]["id"]
    agent_ids = agent_ids.detach().cpu().tolist()
    agent_types_data = batch["agent"].get("type")
    if agent_types_data is not None:
        agent_types = agent_types_data.detach().cpu().tolist()
    else:
        agent_types = [0] * len(agent_ids)
    agent_type_by_id: Dict[str, int] = {
        str(aid): int(at)
        for aid, at in zip(agent_ids, agent_types)
    }
    role = batch["agent"]["role"]
    role_np = role.detach().cpu().numpy()
    ego_idx = [i for i, v in enumerate(role_np) if v[0]]
    ego_id = agent_ids[ego_idx[0]] if ego_idx else None
    ego_type = (
        int(agent_types[ego_idx[0]])
        if ego_idx and ego_idx[0] < len(agent_types)
        else None
    )

    prompts = [""] * len(agent_ids)
    prompt_mask = torch.zeros(len(agent_ids), dtype=torch.bool)
    if scene_id is None:
        return prompts, prompt_mask

    text_mode = str(text_mode).lower()
    text_split = str(text_split).lower()
    bucket = int(scene_id.split("_")[-1]) % 100

    if text_mode in ("full", "text_only"):
        text_root = text_prompt_root / "text_prompts" / text_split / text_prompt_subdir
        text_path = text_root / str(bucket) / f"{scene_id}_10_90_output.txt"
        if text_path.exists():
            sentences = _parse_text_file(text_path)
            for idx, agent_id in enumerate(agent_ids):
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
                        _rewrite_sentence(
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

    if text_mode in ("full", "tag_only"):
        tag_subdir = _infer_tag_subdir(text_split, tag_prompt_subdir)
        tag_root = text_prompt_root / "tag_prompts" / tag_subdir / "tags"
        tag_path = tag_root / str(bucket) / f"{scene_id}.json"
        if tag_path.exists():
            tags = _parse_tag_file(tag_path)
            tag_sentences = _build_tag_sentences(
                tags,
                agent_ids,
                ego_id,
                agent_type_by_id,
                time_scale=tag_time_scale,
                tag_sentence_style=str(tag_sentence_style).lower(),
            )
            for idx, agent_id in enumerate(agent_ids):
                tag_sentence = tag_sentences.get(int(agent_id))
                if not tag_sentence:
                    continue
                if prompts[idx]:
                    prompts[idx] = f"{prompts[idx]} {tag_sentence}"
                else:
                    prompts[idx] = tag_sentence
                prompt_mask[idx] = True

    return prompts, prompt_mask


def _build_control_text_from_custom(
    batch: HeteroData,
    *,
    sentences: List[str],
) -> Tuple[List[str], torch.Tensor]:
    agent_ids = batch["agent"]["id"]
    agent_ids = agent_ids.detach().cpu().tolist()
    agent_types_data = batch["agent"].get("type")
    if agent_types_data is not None:
        agent_types = agent_types_data.detach().cpu().tolist()
    else:
        agent_types = [0] * len(agent_ids)
    agent_type_by_id: Dict[str, int] = {
        str(aid): int(at)
        for aid, at in zip(agent_ids, agent_types)
    }
    role = batch["agent"]["role"]
    role_np = role.detach().cpu().numpy()
    ego_idx = [i for i, v in enumerate(role_np) if v[0]]
    ego_id = agent_ids[ego_idx[0]] if ego_idx else None
    ego_type = (
        int(agent_types[ego_idx[0]])
        if ego_idx and ego_idx[0] < len(agent_types)
        else None
    )

    prompts = [""] * len(agent_ids)
    prompt_mask = torch.zeros(len(agent_ids), dtype=torch.bool)

    if not sentences:
        return prompts, prompt_mask

    for idx, agent_id in enumerate(agent_ids):
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
                _rewrite_sentence(
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

    return prompts, prompt_mask


__all__ = [
    "_load_text_mapping",
    "_build_control_text_for_batch",
    "_parse_custom_text",
    "_build_control_text_from_custom",
]
