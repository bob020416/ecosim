from ecosim.run_inference import INPUT_PATH_KEYS, build_command


def test_vbd_custom_text_command() -> None:
    cmd, cwd = build_command(
        {
            "backend": "vbd",
            "mode": "text",
            "control_type": "custom",
            "paths": {
                "test_path": "data/vbd/validation_pkl",
                "model_path": "checkpoints/vbd/base.ckpt",
                "ctrl_ckpt_path": "checkpoints/vbd/text_control.ckpt",
                "output_dir": "outputs/vbd_text_custom",
            },
            "runtime": {
                "closedloop": True,
                "use_training_format": True,
                "only_controlnet": True,
            },
            "control": {"target_agents": "0"},
            "text": {"custom_text": "<ego> stops."},
        }
    )

    joined = " ".join(cmd)
    assert cwd.name == "VBD"
    assert "custom_text_closedloop.py" in joined
    assert "--custom_text" in cmd
    assert "<ego> stops." in cmd
    assert "--closedloop" in cmd


def test_catk_custom_sketch_command() -> None:
    cmd, cwd = build_command(
        {
            "backend": "catk",
            "mode": "sketch",
            "control_type": "custom",
            "paths": {
                "catk_data_dir": "data/catk/validation",
                "ctrl_ckpt_path": "checkpoints/catk/sketch_control.ckpt",
                "output_dir": "outputs/catk_sketch_custom",
                "custom_sketch_file": "data/catk/custom_sketch/example.json",
            },
            "runtime": {"only_controlnet": True, "save_vis": False},
            "control": {
                "target_agents": "0",
                "cfg_guidance_mode": "manual",
                "cfg_guidance_scale": 0.5,
            },
        }
    )

    joined = " ".join(cmd)
    assert cwd.name == "catk"
    assert "visualize_closedloop.py" in joined
    assert "--sketch" in cmd
    assert "custom" in cmd
    assert "--conditioning_mode" in cmd
    assert "--cfg_guidance_mode" in cmd
    assert "manual" in cmd
    assert "--cfg_guidance_scale" in cmd
    assert "0.5" in cmd
    assert "--no-save_vis" in cmd


def test_check_files_does_not_require_output_dir() -> None:
    assert "output_dir" not in INPUT_PATH_KEYS
