# CATK Deployment

This is the primary inference target for the initial ECoSim release. The CATK
backend supports GT sketch, GT latent, GT text, custom sketch, and custom text
conditioning through one wrapper:

```bash
python -m ecosim.run_inference --config configs/catk_<mode>_gt.yaml
```

## Required Assets

Use this layout, or edit the config paths to absolute locations:

```text
ecosim/
  checkpoints/
    catk/
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
      scenario_lists/
        wosac_1000/
          scenario_<scenario_id>_list.txt
        wosac_1000_text/
          scenario_<scenario_id>_list.txt
      SMART/
        prosim_instruct_520k/
        validation_scenario_to_prosim_scene_strict.json
```

Mode-specific requirements:

| Mode | Config | Required extra assets |
| --- | --- | --- |
| GT sketch | `configs/catk_sketch_gt.yaml` | SMART validation `.pkl` files |
| GT latent | `configs/catk_latent_gt.yaml` | `scenario_<id>_latents.pkl` files |
| GT text | `configs/catk_text_gt.yaml` | ProSIM prompt cache and validation mapping |
| Custom sketch | `configs/catk_sketch_custom.yaml` | custom sketch JSON or GUI display |
| Custom text | `configs/catk_text_custom.yaml` | inline prompt or prompt text file |

`tfrecord_dir` is needed when saving WOSAC-compatible rollouts or richer
visualization metadata.

## Smoke Test

First inspect the command:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_text_gt.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_sketch_custom.yaml --dry-run
python -m ecosim.run_inference --config configs/catk_text_custom.yaml --dry-run
```

After copying assets, check configured input paths:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml --check-files --dry-run
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml --check-files --dry-run
python -m ecosim.run_inference --config configs/catk_text_gt.yaml --check-files --dry-run
```

For a quick functional run, temporarily set these fields in each config:

```yaml
runtime:
  max_scenarios: 1
  wosac_num_rollouts: 1
  save_wosac: true
  save_vis: false
```

Then run:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml
python -m ecosim.run_inference --config configs/catk_text_gt.yaml
```

Expected output files are named like:

```text
outputs/catk_<mode>_gt/<scenario_id>_closedloop_wosac.pkl
```

## Custom Controls

Custom sketch files store local future path points keyed by agent row index:

```json
{
  "agents": {
    "0": [[0.0, 0.0], [8.0, 0.0], [15.0, -4.0], [20.0, -10.0]]
  }
}
```

Custom text follows the same marker-based preprocessing as GT text. Use
`<ego>` for the ego vehicle or `<agent_id>` for an actual Waymo/CATK agent id.
Prompts without one of these markers are ignored by the current parser.

For custom controls, keep `control_mask_mode: sketch`. The option name is
inherited from CATK; it uses the custom sketch/text mask directly instead of
intersecting it with Waymo `tracks_to_predict`.

## Guidance Defaults

CATK uses autoregressive token decoding. The release configs enable logit-level
classifier-free guidance with the paper scales:

| Mode | `cfg_guidance_scale` |
| --- | --- |
| sketch | `0.5` |
| latent | `0.7` |
| text | `1.0` |

Set `cfg_guidance_mode: off` to compare against plain conditional decoding.
