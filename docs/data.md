# Data Layout

The configs assume paths are relative to the repository root. You can also use
absolute paths.

Recommended local layout:

```text
ecosim/
  data/
    vbd/
      validation_pkl/
      latents_behaviorvae/
      prosim_instruct_520k/
      validation_scenario_to_prosim_scene_strict.json
    catk/
      validation/
      validation_tfrecords_splitted/
      latents_catk_4s_wcontext/
      scenario_lists/
        wosac_1000/
        wosac_1000_text/
      SMART/
        prosim_instruct_520k/
        validation_scenario_to_prosim_scene_strict.json
  checkpoints/
    vbd/
    catk/
  outputs/
```

## VBD Inputs

VBD inference expects processed ScenarioNet/VBD `.pkl` scenario files. GT
latent inference additionally expects files named:

```text
scenario_<scenario_id>_latents.pkl
```

Generate these with `third_party/VBD/script/create_vbd_ctrl_latents.py`; see
`docs/behaviorvae.md`.

## CATK Inputs

CATK inference expects SMART/CATK cached `.pkl` files under `catk_data_dir`.
If WOSAC-compatible output or richer visualization is needed, set
`tfrecord_dir` to the split TFRecord directory.

GT latent inference expects files named:

```text
scenario_<scenario_id>_latents.pkl
```

The default CATK latent config looks for these under
`data/catk/latents_catk_4s_wcontext`.

Generate CATK-aligned latent files with
`third_party/catk/scripts/latent/create_catk_latent_pairs.py`; see
`docs/behaviorvae.md`.

GT text inference expects `prosim_instruct_dir` to contain:

```text
SMART/
  prosim_instruct_520k/
  validation_scenario_to_prosim_scene_strict.json
```

## Validation Scenario to ProSim Scene Mapping

The strict validation mapping is included at
[`assets/mappings/validation_scenario_to_prosim_scene_strict.json`](../assets/mappings/validation_scenario_to_prosim_scene_strict.json).
It contains 25,835 entries mapping scenario IDs (JSON keys) to ProSim scene IDs
(JSON string values), for example `"1000a444aa94927d": "scene_24844"`.

To use it with the recommended VBD and CATK data layouts above, copy it into
both data directories:

```bash
mkdir -p data/vbd data/catk/SMART
cp assets/mappings/validation_scenario_to_prosim_scene_strict.json data/vbd/
cp assets/mappings/validation_scenario_to_prosim_scene_strict.json data/catk/SMART/
```

The ProSim instruction dataset itself must be supplied separately.
