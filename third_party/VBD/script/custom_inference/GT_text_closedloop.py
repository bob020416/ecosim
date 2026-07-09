import argparse
from typing import Dict, List, Optional, Tuple

import torch

import utils_closed_loop as utils_cl
import utils as utils_ol
from vbd.data.dataset import WaymaxDataset

"""Closed-loop or open-loop VBD inference with per-scenario GT text conditioning."""

DEFAULT_TEXT_PROMPT_ROOT = "./data/prosim_instruct_520k"
DEFAULT_TEXT_MAPPING_PATH = "./data/validation_scenario_to_prosim_scene_strict.json"
DEFAULT_TEXT_PROMPT_SUBDIR = "70b_0.8_0.9_action_prompt_v2"
DEFAULT_TEXT_SPLIT = "val"
DEFAULT_TEXT_MODE = "full"
DEFAULT_TAG_PROMPT_SUBDIR = "auto"
DEFAULT_TAG_SENTENCE_STYLE = "ordered"
DEFAULT_TAG_TIME_SCALE = 0.1
DEFAULT_ANCHOR_PATH = "./vbd/data/cluster_64_center_dict.pkl"


def _load_text_cfg_from_ckpt(ckpt_path: str) -> dict:
    if not ckpt_path:
        return {}
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        print(f"[GT_text_closedloop] Warning: failed to load checkpoint {ckpt_path}: {exc}")
        return {}

    cfg = None
    if isinstance(checkpoint, dict):
        if "hyper_parameters" in checkpoint and isinstance(checkpoint["hyper_parameters"], dict):
            cfg = checkpoint["hyper_parameters"].get("cfg", None)
        if cfg is None and "cfg" in checkpoint:
            cfg = checkpoint.get("cfg", None)
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _no_sketch(batch):
    return None, []


class TextPromptAttacher:
    def __init__(
        self,
        text_prompt_root: str,
        text_mapping_path: str,
        text_prompt_subdir: str,
        text_split: str,
        text_mode: str,
        tag_prompt_root: Optional[str],
        tag_prompt_subdir: str,
        tag_sentence_style: str,
        tag_time_scale: float,
        filter_text_mapping: bool,
        filter_text_prompts: bool,
        debug_text_mapping: bool,
        strict_text: bool,
        print_samples: int,
    ) -> None:
        if tag_prompt_root is None:
            tag_prompt_root = text_prompt_root

        self.dataset = WaymaxDataset(
            data_dir=None,
            anchor_path=DEFAULT_ANCHOR_PATH,
            conditioning_mode="text",
            text_prompt_root=text_prompt_root,
            text_mapping_path=text_mapping_path,
            text_prompt_subdir=text_prompt_subdir,
            text_split=text_split,
            text_mode=text_mode,
            tag_prompt_root=tag_prompt_root,
            tag_prompt_subdir=tag_prompt_subdir,
            tag_time_scale=tag_time_scale,
            tag_sentence_style=tag_sentence_style,
            attach_text_prompts=True,
            filter_text_mapping=filter_text_mapping,
            filter_text_prompts=filter_text_prompts,
            debug_text_mapping=debug_text_mapping,
        )

        self.strict_text = bool(strict_text)
        self.print_samples = max(0, int(print_samples))
        self.samples_printed = 0
        self.filter_text_mapping = bool(filter_text_mapping)
        self.filter_text_prompts = bool(filter_text_prompts)
        self.debug_text_mapping = bool(debug_text_mapping)
        self.stats: Dict[str, int] = {
            "total": 0,
            "missing_mapping": 0,
            "missing_text": 0,
            "missing_tag": 0,
            "empty_prompt": 0,
        }

    def _warn_or_raise(self, msg: str) -> None:
        if self.strict_text:
            raise RuntimeError(msg)
        print(f"[GT_text_closedloop] Warning: {msg}")

    def _check_prompt_files(self, scenario_id: str, scene_id: str) -> None:
        missing_text = False
        missing_tag = False

        if self.dataset._text_mode in ("full", "text_only") and self.dataset._text_prompt_root is not None:
            split = self.dataset._infer_text_split()
            text_root = (
                self.dataset._text_prompt_root
                / "text_prompts"
                / split
                / self.dataset._text_prompt_subdir
            )
            text_path = (
                text_root
                / str(self.dataset._scene_id_to_bucket(scene_id))
                / f"{scene_id}_10_90_output.txt"
            )
            if not text_path.exists():
                missing_text = True
                self.stats["missing_text"] += 1

        if self.dataset._text_mode in ("full", "tag_only") and self.dataset._tag_prompt_root is not None:
            tag_root = (
                self.dataset._tag_prompt_root
                / "tag_prompts"
                / self.dataset._infer_tag_subdir()
                / "tags"
            )
            tag_path = (
                tag_root
                / str(self.dataset._scene_id_to_bucket(scene_id))
                / f"{scene_id}.json"
            )
            if not tag_path.exists():
                missing_tag = True
                self.stats["missing_tag"] += 1

        if missing_text or missing_tag:
            self._warn_or_raise(
                f"Missing prompt files for scenario_id={scenario_id} scene_id={scene_id} "
                f"text={'ok' if not missing_text else 'missing'} "
                f"tag={'ok' if not missing_tag else 'missing'}"
            )

    def _format_examples(self, scenario_data, prompts, mask, max_examples: int = 3) -> List[str]:
        agent_ids = scenario_data.get("agents_id")
        if agent_ids is None:
            agent_ids = scenario_data.get("agents_object_id")
        if hasattr(agent_ids, "tolist"):
            agent_ids_list = agent_ids.tolist()
        elif agent_ids is not None:
            agent_ids_list = list(agent_ids)
        else:
            agent_ids_list = list(range(len(prompts)))

        examples: List[str] = []
        for idx, prompt in enumerate(prompts):
            if idx >= len(agent_ids_list):
                break
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            if isinstance(mask, torch.Tensor):
                if idx >= mask.numel() or not bool(mask[idx].item()):
                    continue
            display = prompt.strip().replace("\n", " ")
            if len(display) > 160:
                display = display[:160] + "..."
            examples.append(f"agent {agent_ids_list[idx]}: {display}")
            if len(examples) >= max_examples:
                break
        return examples

    def attach(self, batch: dict, scenario_data: dict) -> None:
        self.stats["total"] += 1
        scenario_id = scenario_data.get("scenario_id")
        if not scenario_id:
            raise RuntimeError("Missing scenario_id in scenario_data")

        scene_id = None
        if self.dataset._text_mapping is not None:
            scene_id = self.dataset._text_mapping.get(scenario_id)

        if scene_id is None:
            self.stats["missing_mapping"] += 1
            self._warn_or_raise(f"Missing text mapping for scenario_id={scenario_id}")
        else:
            self._check_prompt_files(scenario_id, scene_id)

        # Attach prompts exactly as training does.
        self.dataset._attach_text_prompts(scenario_data, batch)

        mask = batch.get("text_prompt_mask")
        prompts = batch.get("text_prompt", [])
        if isinstance(mask, torch.Tensor):
            valid = int(mask.sum().item())
            total = int(mask.numel())
        else:
            valid = sum(1 for p in prompts if isinstance(p, str) and p.strip())
            total = len(prompts)

        if valid == 0:
            self.stats["empty_prompt"] += 1

        if self.print_samples and self.samples_printed < self.print_samples:
            self.samples_printed += 1
            examples = self._format_examples(scenario_data, prompts, mask)
            print(
                f"[GT_text_closedloop] scenario_id={scenario_id} scene_id={scene_id} "
                f"prompt_valid={valid}/{total}"
            )
            for line in examples:
                print(f"[GT_text_closedloop]   {line}")

    def print_config(self) -> None:
        print("[GT_text_closedloop] Text prompt config:")
        print(f"  text_prompt_root: {self.dataset._text_prompt_root}")
        print(f"  text_mapping_path: {self.dataset._text_mapping is not None}")
        print(f"  text_prompt_subdir: {self.dataset._text_prompt_subdir}")
        print(f"  text_split: {self.dataset._text_split}")
        print(f"  text_mode: {self.dataset._text_mode}")
        print(f"  tag_prompt_root: {self.dataset._tag_prompt_root}")
        print(f"  tag_prompt_subdir: {self.dataset._tag_prompt_subdir}")
        print(f"  tag_sentence_style: {self.dataset._tag_sentence_style}")
        print(f"  tag_time_scale: {self.dataset._tag_time_scale}")
        print(f"  filter_text_mapping: {self.filter_text_mapping}")
        print(f"  filter_text_prompts: {self.filter_text_prompts}")
        print(f"  debug_text_mapping: {self.debug_text_mapping}")
        print(f"  strict_text: {self.strict_text}")
        print(f"  print_samples: {self.print_samples}")

    def print_summary(self) -> None:
        print("[GT_text_closedloop] Text prompt summary:")
        for key in ["total", "missing_mapping", "missing_text", "missing_tag", "empty_prompt"]:
            print(f"  {key}: {self.stats.get(key, 0)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_path", type=str, required=True, help="Directory containing PKL files")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ctrl_ckpt_path", type=str, default=None)
    parser.add_argument("--test_mode", type=str, default="diffusion", choices=["diffusion", "prior"])
    parser.add_argument("--save_simulation", action="store_true")
    parser.add_argument("--plot_control_samples", action="store_true")
    parser.add_argument("--use_training_format", action="store_true", default=True)
    parser.add_argument("--max_scenarios", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="test_closedloop_gt_text")
    parser.add_argument("--only_controlnet", action="store_true", help="Only run ControlNet (skip baseline) for speed.")
    parser.add_argument("--closedloop", action="store_true")
    parser.add_argument("--horizon_steps", type=int, default=11)
    parser.add_argument("--total_steps", type=int, default=80)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=32)

    parser.add_argument(
        "--auto_text_config",
        action="store_true",
        help="Load text prompt settings (text_mode/prompt paths/filters) from checkpoint config.",
    )
    parser.add_argument("--text_prompt_root", type=str, default=DEFAULT_TEXT_PROMPT_ROOT)
    parser.add_argument("--text_mapping_path", type=str, default=DEFAULT_TEXT_MAPPING_PATH)
    parser.add_argument("--text_prompt_subdir", type=str, default=DEFAULT_TEXT_PROMPT_SUBDIR)
    parser.add_argument("--text_split", type=str, default=DEFAULT_TEXT_SPLIT)
    parser.add_argument("--text_mode", type=str, default=DEFAULT_TEXT_MODE, choices=["full", "text_only", "tag_only"])
    parser.add_argument("--tag_prompt_root", type=str, default=None)
    parser.add_argument("--tag_prompt_subdir", type=str, default=DEFAULT_TAG_PROMPT_SUBDIR)
    parser.add_argument("--tag_sentence_style", type=str, default=DEFAULT_TAG_SENTENCE_STYLE)
    parser.add_argument("--tag_time_scale", type=float, default=DEFAULT_TAG_TIME_SCALE)
    parser.add_argument(
        "--text_filter_mapping",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override text_filter_mapping (true/false).",
    )
    parser.add_argument(
        "--text_filter_prompts",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override text_filter_prompts (true/false).",
    )
    parser.add_argument(
        "--text_filter_debug",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override text_filter_debug (true/false).",
    )
    parser.add_argument(
        "--allow_missing_text",
        action="store_true",
        help="Do not fail fast on missing mapping/prompt files.",
    )
    parser.add_argument(
        "--print_text_samples",
        type=int,
        default=3,
        help="Print the first N scenarios' prompts for sanity check (0 disables).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    text_cfg = {}
    if args.auto_text_config:
        ckpt_path = args.ctrl_ckpt_path or args.model_path
        text_cfg = _load_text_cfg_from_ckpt(ckpt_path)
        # Override text-related settings from checkpoint config when available.
        args.text_prompt_root = text_cfg.get("text_prompt_root", args.text_prompt_root)
        args.text_mapping_path = text_cfg.get("text_mapping_path", args.text_mapping_path)
        args.text_prompt_subdir = text_cfg.get("text_prompt_subdir", args.text_prompt_subdir)
        args.text_split = text_cfg.get("text_split", args.text_split)
        args.text_mode = text_cfg.get("text_mode", args.text_mode)
        args.tag_prompt_root = text_cfg.get("tag_prompt_root", args.tag_prompt_root)
        args.tag_prompt_subdir = text_cfg.get("tag_prompt_subdir", args.tag_prompt_subdir)
        args.tag_sentence_style = text_cfg.get("tag_sentence_style", args.tag_sentence_style)
        args.tag_time_scale = text_cfg.get("tag_time_scale", args.tag_time_scale)

    filter_mapping_override = _str_to_bool(args.text_filter_mapping)
    filter_prompts_override = _str_to_bool(args.text_filter_prompts)
    filter_debug_override = _str_to_bool(args.text_filter_debug)
    filter_text_mapping = (
        filter_mapping_override
        if filter_mapping_override is not None
        else bool(text_cfg.get("text_filter_mapping", False))
    )
    filter_text_prompts = (
        filter_prompts_override
        if filter_prompts_override is not None
        else bool(text_cfg.get("text_filter_prompts", False))
    )
    debug_text_mapping = (
        filter_debug_override
        if filter_debug_override is not None
        else bool(text_cfg.get("text_filter_debug", False))
    )

    attacher = TextPromptAttacher(
        text_prompt_root=args.text_prompt_root,
        text_mapping_path=args.text_mapping_path,
        text_prompt_subdir=args.text_prompt_subdir,
        text_split=args.text_split,
        text_mode=args.text_mode,
        tag_prompt_root=args.tag_prompt_root,
        tag_prompt_subdir=args.tag_prompt_subdir,
        tag_sentence_style=args.tag_sentence_style,
        tag_time_scale=args.tag_time_scale,
        filter_text_mapping=filter_text_mapping,
        filter_text_prompts=filter_text_prompts,
        debug_text_mapping=debug_text_mapping,
        strict_text=not args.allow_missing_text,
        print_samples=args.print_text_samples,
    )
    attacher.print_config()

    if args.closedloop:
        print("\nRunning CLOSED-LOOP simulation with GT text conditioning")
        utils_cl.run_closedloop_simulation(
            args,
            _no_sketch,
            sketch_name="GT Text",
            batch_preprocess_fn=attacher.attach,
        )
    else:
        print("\nRunning OPEN-LOOP simulation with GT text conditioning")
        utils_ol.run_simulation(
            args,
            _no_sketch,
            sketch_name="GT Text",
            batch_preprocess_fn=attacher.attach,
        )

    attacher.print_summary()


if __name__ == "__main__":
    main()
