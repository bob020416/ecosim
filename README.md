# ECoSim

**ECoSim: Data Efficient Fine-Tuning for Controllable Traffic Simulation**  
ECCV 2026

Yu-Hsiang Chen*, Wei-Jer Chang*, Yi-Ting Chen, Masayoshi Tomizuka  
*Equal contribution*

ECoSim adds lightweight control adapters to frozen traffic simulation backbones
for controllable closed-loop traffic generation. This repository is currently
organized as an **inference-first release** for VBD and CATK/SMART, with CATK
as the primary validated deployment target.

Large datasets, trained checkpoints, and generated rollout files are not stored
in git.

## Release Status

CATK inference is the current supported path:

- [x] CATK GT sketch inference
- [x] CATK GT latent inference
- [x] CATK GT text-prompt inference
- [x] CATK custom sketch inference from JSON or interactive drawing
- [x] CATK custom text inference from inline prompts or prompt files
- [x] CATK autoregressive logit-level classifier-free guidance
- [x] BehaviorVAE training and latent export utilities
- [x] Config-driven wrapper for VBD and CATK inference
- [ ] Clean public training pipeline for ECoSim control adapters
- [X] Hosted checkpoint and data download links
- [ ] Final license and third-party redistribution review

The missing training item above refers to the public, end-to-end adapter
training workflow. Inference, latent consumption, and BehaviorVAE latent export
utilities are already included.

## Installation

Use the backend environment you plan to run. For CATK inference, activate the
CATK/SMART environment, then install the lightweight wrapper:

```bash
pip install -e .
```

For VBD or BehaviorVAE utilities, use the VBD-compatible environment.

## Required Assets

The configs use repository-relative paths by default. You can also replace any
path with an absolute path.

Initial Released of each modality's checkpoing can be download here:
https://drive.google.com/drive/folders/12EPwbLdtYCR_zafChCksOJ0qxbZTXkz4?usp=sharing

```text
ecosim/
  checkpoints/
    catk/
      sketch_control.ckpt
      latent_control.ckpt
      text_control.ckpt
    vbd/
      base.ckpt
      sketch_control.ckpt
      latent_control.ckpt
      text_control.ckpt
  data/
    catk/
      validation/
        <scenario_id>.pkl
      validation_tfrecords_splitted/
        <scenario_id>.tfrecords
      latents_catk_4s_wcontext/
        scenario_<scenario_id>_latents.pkl
      SMART/
        prosim_instruct_520k/
        validation_scenario_to_prosim_scene_strict.json
```

More details are in [docs/data.md](docs/data.md) and
[docs/checkpoints.md](docs/checkpoints.md).

## CATK Quick Start

Inspect the exact command without running inference:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_text_gt.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_sketch_custom.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_text_custom.yaml --dry-run
```

Check that configured input files exist:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml --check-files --dry-run
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml --check-files --dry-run
python -m ecosim.run_inference --config configs/catk_text_gt.yaml --check-files --dry-run
```

Run inference:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml
python -m ecosim.run_inference --config configs/catk_text_gt.yaml
```

Expected WOSAC rollout outputs are named like:

```text
outputs/catk_<mode>_gt/<scenario_id>_closedloop_wosac.pkl
```

For a fast smoke test, set this in the config:

```yaml
runtime:
  max_scenarios: 1
  wosac_num_rollouts: 1
  save_wosac: true
  save_vis: false
```

## CATK Configs

| Control | Config | Main checkpoint | Extra input |
| --- | --- | --- | --- |
| GT sketch | `configs/catk_sketch_gt.yaml` | `checkpoints/catk/sketch_control.ckpt` | CATK validation scenarios |
| GT latent | `configs/catk_latent_gt.yaml` | `checkpoints/catk/latent_control.ckpt` | `scenario_<id>_latents.pkl` |
| GT text | `configs/catk_text_gt.yaml` | `checkpoints/catk/text_control.ckpt` | ProSIM prompt cache |
| Custom sketch | `configs/catk_sketch_custom.yaml` | `checkpoints/catk/sketch_control.ckpt` | custom sketch JSON or GUI drawing |
| Custom text | `configs/catk_text_custom.yaml` | `checkpoints/catk/text_control.ckpt` | inline prompt or prompt file |

The deployment checklist is in
[docs/catk_deployment.md](docs/catk_deployment.md).

## Custom Sketch

Use `configs/catk_sketch_custom.yaml`:

```yaml
paths:
  custom_sketch_file: examples/custom_sketch/example.json

control:
  sketch: custom
  target_agents: "0"
  control_mask_mode: sketch
```

The JSON file stores local future path points per agent row index:

```json
{
  "agents": {
    "0": [[0.0, 0.0], [8.0, 0.0], [15.0, -4.0], [20.0, -10.0]]
  }
}
```

Coordinates are in the controlled agent's local frame at the current planning
step: `x` is forward and `y` is lateral. If `custom_sketch_file` is omitted,
the script opens an interactive drawing window; headless servers should use the
JSON file path.

For custom controls, keep `control_mask_mode: sketch`. The option name comes
from the CATK script; in this release it means the custom sketch/text mask is
used directly instead of intersecting it with Waymo `tracks_to_predict`.

## Custom Text

Use `configs/catk_text_custom.yaml`:

```yaml
runtime:
  debug_print_prompts: true

text:
  custom_text: "<ego> is decelerating and coming to a stop."
  custom_text_delim: "||"

control:
  control_mask_mode: sketch
```

Multiple prompts can be separated with `custom_text_delim`:

```yaml
text:
  custom_text: "<ego> is turning right.||<ego> is accelerating."
```

Or use a text file with one prompt per line:

```yaml
paths:
  custom_text_file: examples/custom_text/prompts.txt
```

Current custom text prompts follow the same marker-based preprocessing as GT
text:

- Use `<ego>` for the ego vehicle.
- Use `<agent_id>` for an actual Waymo/CATK agent id.
- Prompts without `<ego>` or `<agent_id>` markers are ignored.
- `debug_print_prompts: true` prints the final per-agent prompt after rewriting.

For example, `<ego> is turning right.` is rewritten to the model-facing style
`the target vehicle is turning right.` when ego is the controlled agent.

## Latent Control

GT latent configs consume precomputed BehaviorVAE latent files. To generate
them from VBD-format scenarios:

```bash
cd third_party/catk
python scripts/latent/create_catk_latent_pairs.py \
  --dataset_root ../VBD/data/waymovalid-scenarionet \
  --ckpt ../../checkpoints/vbd/behaviorvae.ckpt \
  --cfg ../VBD/config/BehaviorVAE_full.yaml \
  --latent_dir_name latents_catk_4s_wcontext \
  --stride 5 \
  --pred_h 40 \
  --num_steps 16
```

See [docs/behaviorvae.md](docs/behaviorvae.md) for BehaviorVAE training and
VBD/CATK latent export.

## Classifier-Free Guidance

CATK/SMART uses autoregressive trajectory token decoding, so ECoSim applies
classifier-free guidance directly on next-token logits:

```text
L_final = L_base + omega * (L_control - L_base)
```

The public CATK configs use the paper scales:

| Modality | `cfg_guidance_scale` |
| --- | --- |
| sketch | `0.5` |
| latent | `0.7` |
| text | `1.0` |

Set `cfg_guidance_mode: off` to compare against plain conditional decoding.

## Repository Layout

```text
ecosim/
  ecosim/                  # config-driven runner
  configs/                 # public inference examples
  docs/                    # setup, data, checkpoint, and pipeline notes
  examples/                # custom sketch/text examples
  scripts/infer.py         # thin script entry point
  third_party/VBD/         # vendored VBD subset used by ECoSim
  third_party/catk/        # vendored CATK/SMART subset used by ECoSim
```

## Citation

```bibtex
@inproceedings{chen2026ecosim,
  title = {ECoSim: Data Efficient Fine-Tuning for Controllable Traffic Simulation},
  author = {Chen, Yu-Hsiang and Chang, Wei-Jer and Chen, Yi-Ting and Tomizuka, Masayoshi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2026}
}
```

## Acknowledgements

This code builds on VBD and CATK/SMART. See `third_party/NOTICE.md` and the
upstream README files under each vendored backend.
