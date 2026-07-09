# Checkpoints

Model weights are not committed. Put them under `checkpoints/` or point configs
to absolute paths.

Recommended layout:

```text
checkpoints/
  vbd/
    base.ckpt
    sketch_control.ckpt
    latent_control.ckpt
    text_control.ckpt
    behaviorvae.ckpt
  catk/
    base.ckpt
    sketch_control.ckpt
    latent_control.ckpt
    text_control.ckpt
```

Each inference config needs:

- `model_path` and `ctrl_ckpt_path` for VBD.
- `base_ckpt_path`, `ctrl_ckpt_path`, or both for CATK.

For normal ECoSim controlled inference, provide `ctrl_ckpt_path` and set
`only_controlnet: true`.

BehaviorVAE latent export additionally needs a BehaviorVAE checkpoint, usually
paired with the config used for training. If `config.yaml` is not stored next
to the checkpoint, pass `--cfg third_party/VBD/config/BehaviorVAE_full.yaml`
or another explicit config path to the latent export scripts.

The initial CATK deployment target needs:

- `checkpoints/catk/sketch_control.ckpt`
- `checkpoints/catk/latent_control.ckpt`
- `checkpoints/catk/text_control.ckpt`
