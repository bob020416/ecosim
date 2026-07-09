from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"

PATH_KEYS = {
    "base_ckpt_path",
    "catk_data_dir",
    "ctrl_ckpt_path",
    "custom_sketch_file",
    "custom_text_file",
    "latent_dir",
    "latent_path",
    "model_path",
    "output_dir",
    "prosim_instruct_dir",
    "scenario_id_dir",
    "tag_prompt_root",
    "test_path",
    "text_mapping_path",
    "text_prompt_root",
    "tfrecord_dir",
}
INPUT_PATH_KEYS = PATH_KEYS - {"output_dir"}

VBD_SCRIPT_BY_MODE = {
    ("sketch", "gt"): "GT_sketch_closedloop.py",
    ("sketch", "custom"): "custom_sketch_closedloop.py",
    ("sketch", "forward"): "custom_sketch_closedloop.py",
    ("sketch", "right_turn"): "custom_sketch_closedloop.py",
    ("sketch", "static"): "custom_sketch_closedloop.py",
    ("latent", "gt"): "GT_latent_closedloop.py",
    ("text", "gt"): "GT_text_closedloop.py",
    ("text", "custom"): "custom_text_closedloop.py",
}


def _load_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "YAML config support requires PyYAML. Install it or provide a JSON config."
        ) from exc
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Config root must be a mapping.")
        return data
    return _load_yaml(text)


def _section(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return dict(value)


def _params(config: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for name in ("paths", "runtime", "control", "text"):
        merged.update(_section(config, name))
    return merged


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _resolve_path(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(_resolve_path(item)) for item in value]
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _value_for_arg(key: str, value: Any) -> Any:
    if key in PATH_KEYS:
        return _resolve_path(value)
    return value


def _add_value(cmd: List[str], name: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{name}"
    value = _value_for_arg(name, value)
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            cmd.extend([flag, str(item)])
        return
    cmd.extend([flag, str(value)])


def _add_store_true(cmd: List[str], name: str, value: Any) -> None:
    if value is not None and _as_bool(value):
        cmd.append(f"--{name}")


def _add_boolean_optional(cmd: List[str], name: str, value: Any) -> None:
    if value is None:
        return
    cmd.append(f"--{name}" if _as_bool(value) else f"--no-{name}")


def _append_extra_args(cmd: List[str], extra_args: Any) -> None:
    if not extra_args:
        return
    if isinstance(extra_args, str):
        cmd.extend(shlex.split(extra_args))
        return
    if not isinstance(extra_args, Sequence):
        raise ValueError("extra_args must be a list or a shell-style string.")
    cmd.extend(str(item) for item in extra_args)


def _require(params: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if params.get(key) in (None, "")]
    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))


def _check_files(params: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = []
    for key in keys:
        value = params.get(key)
        if value in (None, ""):
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            resolved = Path(str(_value_for_arg(key, item)))
            if not resolved.exists():
                missing.append(f"{key}={resolved}")
    if missing:
        raise FileNotFoundError("Missing required files/directories:\n  " + "\n  ".join(missing))


def _python_bin(config: Mapping[str, Any]) -> str:
    return str(config.get("python", sys.executable))


def _control_type(config: Mapping[str, Any], params: Mapping[str, Any]) -> str:
    raw = config.get("control_type", params.get("control_type", "gt"))
    return str(raw).strip().lower()


def build_vbd_command(config: Mapping[str, Any]) -> Tuple[List[str], Path]:
    params = _params(config)
    mode = str(config.get("mode", params.get("mode", "sketch"))).strip().lower()
    control_type = _control_type(config, params)
    script_name = VBD_SCRIPT_BY_MODE.get((mode, control_type))
    if script_name is None:
        raise ValueError(
            f"Unsupported VBD mode/control_type: mode={mode!r}, control_type={control_type!r}"
        )

    root = THIRD_PARTY_ROOT / "VBD"
    script = root / "script" / "custom_inference" / script_name
    cmd = [_python_bin(config), str(script)]

    _require(params, ["test_path", "model_path", "ctrl_ckpt_path", "output_dir"])
    if mode == "latent":
        _require(params, ["latent_path"])
    if mode == "text" and control_type == "custom":
        _require(params, ["custom_text"])

    for name in ("test_path", "model_path", "ctrl_ckpt_path", "output_dir", "latent_path"):
        _add_value(cmd, name, params.get(name))

    for name in (
        "seed",
        "device",
        "test_mode",
        "max_scenarios",
        "horizon_steps",
        "total_steps",
        "num_candidates",
        "batch_size",
        "num_samples",
    ):
        _add_value(cmd, name, params.get(name))

    for name in (
        "save_simulation",
        "plot_control_samples",
        "use_training_format",
        "only_controlnet",
        "closedloop",
    ):
        _add_store_true(cmd, name, params.get(name))

    if mode == "sketch" and control_type != "gt":
        sketch = params.get("sketch", control_type)
        if sketch == "custom":
            sketch = params.get("template", "right_turn")
        _add_value(cmd, "sketch", sketch)
        for name in (
            "target_agents",
            "forward_distance",
            "turn_distance",
            "turn_angle_deg",
        ):
            _add_value(cmd, name, params.get(name))

    if mode == "text":
        if control_type == "gt":
            for name in (
                "auto_text_config",
                "allow_missing_text",
            ):
                _add_store_true(cmd, name, params.get(name))
            for name in (
                "text_prompt_root",
                "text_mapping_path",
                "text_prompt_subdir",
                "text_split",
                "text_mode",
                "tag_prompt_root",
                "tag_prompt_subdir",
                "tag_sentence_style",
                "tag_time_scale",
                "text_filter_mapping",
                "text_filter_prompts",
                "text_filter_debug",
                "print_text_samples",
            ):
                _add_value(cmd, name, params.get(name))
        else:
            for name in ("custom_text", "custom_text_delim", "target_agents", "print_text_samples"):
                _add_value(cmd, name, params.get(name))

    _append_extra_args(cmd, config.get("extra_args"))
    return cmd, root


def build_catk_command(config: Mapping[str, Any]) -> Tuple[List[str], Path]:
    params = _params(config)
    mode = str(config.get("mode", params.get("mode", "sketch"))).strip().lower()
    control_type = _control_type(config, params)
    root = THIRD_PARTY_ROOT / "catk"
    script = root / "scripts" / "custom_inference" / "visualize_closedloop.py"
    cmd = [_python_bin(config), str(script)]

    _require(params, ["catk_data_dir", "output_dir"])
    if not params.get("base_ckpt_path") and not params.get("ctrl_ckpt_path"):
        raise ValueError("CATK requires at least one of base_ckpt_path or ctrl_ckpt_path.")
    if mode == "latent":
        _require(params, ["latent_dir"])
    if mode == "text" and control_type == "gt":
        _require(params, ["prosim_instruct_dir"])
    if mode == "text" and control_type == "custom":
        if not params.get("custom_text") and not params.get("custom_text_file"):
            raise ValueError("CATK custom text requires custom_text or custom_text_file.")

    for name in (
        "base_ckpt_path",
        "ctrl_ckpt_path",
        "catk_data_dir",
        "tfrecord_dir",
        "output_dir",
        "scenario_id_dir",
        "latent_dir",
        "custom_sketch_file",
        "custom_text_file",
        "prosim_instruct_dir",
    ):
        _add_value(cmd, name, params.get(name))

    scenario_ids = params.get("scenario_id")
    if scenario_ids:
        if isinstance(scenario_ids, str):
            scenario_ids = [scenario_ids]
        cmd.append("--scenario_id")
        cmd.extend(str(item) for item in scenario_ids)

    for name in (
        "max_scenarios",
        "seed",
        "device",
        "forward_distance",
        "turn_distance",
        "turn_angle_deg",
        "target_agents",
        "custom_sketch_view_range",
        "custom_sketch_min_points",
        "custom_text",
        "custom_text_delim",
        "text_prompt_subdir",
        "text_mode",
        "text_split",
        "tag_prompt_subdir",
        "tag_sentence_style",
        "tag_time_scale",
        "control_mask_mode",
        "control_max_num",
        "wosac_num_rollouts",
        "latent_target_agents",
        "cfg_guidance_mode",
        "cfg_guidance_scale",
    ):
        _add_value(cmd, name, params.get(name))

    for name in (
        "only_controlnet",
        "custom_sketch_save_inputs",
        "debug_print_prompts",
        "save_wosac",
        "save_vis",
    ):
        _add_boolean_optional(cmd, name, params.get(name))

    if mode == "sketch":
        sketch = control_type
        if sketch == "custom":
            sketch = params.get("sketch", "custom")
        _add_value(cmd, "sketch", sketch)
        _add_value(cmd, "conditioning_mode", params.get("conditioning_mode", "sketch"))
    elif mode == "latent":
        _add_value(cmd, "latent", control_type if control_type in {"gt", "custom"} else "gt")
        _add_value(cmd, "conditioning_mode", params.get("conditioning_mode", "latent"))
        _add_value(cmd, "sketch", params.get("sketch"))
    elif mode == "text":
        _add_value(cmd, "text", control_type)
        if params.get("conditioning_mode"):
            _add_value(cmd, "conditioning_mode", params.get("conditioning_mode"))
    else:
        raise ValueError(f"Unsupported CATK mode: {mode!r}")

    _append_extra_args(cmd, config.get("extra_args"))
    return cmd, root


def build_command(config: Mapping[str, Any]) -> Tuple[List[str], Path]:
    backend = str(config.get("backend", "")).strip().lower()
    if backend == "vbd":
        return build_vbd_command(config)
    if backend == "catk":
        return build_catk_command(config)
    raise ValueError("Config key 'backend' must be either 'vbd' or 'catk'.")


def build_env(cwd: Path) -> MutableMapping[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(cwd)]
    if cwd.name == "catk":
        entries.append(str(cwd / "src"))
    entries.append(str(REPO_ROOT))
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def command_to_string(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ECoSim inference from a YAML/JSON config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Check configured file/directory paths before launching inference.",
    )
    args = parser.parse_args(argv)

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = load_config(config_path)
    cmd, cwd = build_command(config)

    if args.check_files:
        params = _params(config)
        _check_files(params, INPUT_PATH_KEYS)

    print(f"[ECoSim] cwd: {cwd}")
    print(f"[ECoSim] command: {command_to_string(cmd)}")
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=str(cwd), env=build_env(cwd), check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
