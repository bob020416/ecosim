# BehaviorVAE Latent Pipeline

BehaviorVAE is used to encode future agent behavior into latent controls for
the latent-conditioned VBD and CATK adapters. The public inference configs can
consume precomputed latent files, and this release now includes the scripts
needed to train BehaviorVAE and export those files.

Large BehaviorVAE checkpoints and generated latent `.pkl` files are not stored
in git.

## Train BehaviorVAE

Run from the VBD backend directory so the default config paths match the
original training layout:

```bash
cd third_party/VBD
python script/train_behavior_vae.py \
  -cfg config/BehaviorVAE_full.yaml \
  --train_data_path data/waymovalid-scenarionet/training/vbd \
  --val_data_path data/waymovalid-scenarionet/testing/vbd \
  --epochs 200 \
  --num_gpus 1
```

The default config is:

```text
third_party/VBD/config/BehaviorVAE_full.yaml
```

The trained checkpoint should be saved under `train_log_behavior_vae_full/`
unless `log_dir` is overridden.

## Export VBD Latents

VBD closed-loop latent inference expects files named:

```text
scenario_<scenario_id>_latents.pkl
```

with `agent_latents_segments` shaped `[S, A, Z]`. Generate them with:

```bash
cd third_party/VBD
python script/create_vbd_ctrl_latents.py \
  --dataset_root data/waymovalid-scenarionet \
  --ckpt ../../checkpoints/vbd/behaviorvae.ckpt \
  --cfg config/BehaviorVAE_full.yaml \
  --latent_dir_name latents_behaviorvae \
  --exec_h 11 \
  --pred_h 40
```

The VBD inference config consumes the output through:

```yaml
paths:
  latent_path: data/vbd/latents_behaviorvae
```

## Export CATK Latents

CATK/SMART latent inference expects files named:

```text
scenario_<scenario_id>_latents.pkl
```

with `agent_latents` shaped `[S, A, Z]`, where the steps are aligned to CATK
autoregressive decoding. Generate them with:

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

The CATK inference config consumes the output through:

```yaml
paths:
  latent_dir: data/catk/latents_catk_4s_wcontext
```

## Notes

- Use `--cfg auto` only when `config.yaml` is stored next to the BehaviorVAE
  checkpoint. Otherwise pass the config path explicitly.
- If the checkpoint config contains a relative `anchor_path`, the extraction
  scripts try the checkpoint config directory, the vendored VBD root, and the
  current working directory.
- Use deterministic posterior means by default. Add `--sample_latent` only when
  stochastic latent sampling is desired.
