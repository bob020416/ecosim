# Inference

All examples run through one wrapper:

```bash
python -m ecosim.run_inference --config <config.yaml>
```

Use `--dry-run` first to inspect the command without loading data or checkpoints:

```bash
python -m ecosim.run_inference --config configs/vbd_text_custom.yaml --dry-run
```

Use `--check-files` after assets are downloaded. Add `--dry-run` when you only
want to validate paths and inspect the command:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml --check-files --dry-run
```

## VBD

GT sketch:

```bash
python -m ecosim.run_inference --config configs/vbd_sketch_gt.yaml
```

Custom sketch:

```bash
python -m ecosim.run_inference --config configs/vbd_sketch_custom.yaml
```

GT latent:

```bash
python -m ecosim.run_inference --config configs/vbd_latent_gt.yaml
```

GT text or custom text:

```bash
python -m ecosim.run_inference --config configs/vbd_text_gt.yaml
python -m ecosim.run_inference --config configs/vbd_text_custom.yaml
```

## CATK

For a deployment-oriented checklist, see `docs/catk_deployment.md`.

GT/custom sketch:

```bash
python -m ecosim.run_inference --config configs/catk_sketch_gt.yaml
python -m ecosim.run_inference --config configs/catk_sketch_custom.yaml
```

CATK custom sketch can use a JSON file with local points per agent row index.
See `examples/custom_sketch/example.json`.

GT latent:

```bash
python -m ecosim.run_inference --config configs/catk_latent_gt.yaml
```

GT/custom text:

```bash
python -m ecosim.run_inference --config configs/catk_text_gt.yaml
python -m ecosim.run_inference --config configs/catk_text_custom.yaml
```

CATK custom text currently requires `<ego>` or `<agent_id>` markers. Prompts
without a marker are ignored by the parser. Use `debug_print_prompts: true` in
the config to print the final per-agent prompts.

### CATK classifier-free guidance

CATK/SMART uses autoregressive trajectory token decoding, so ECoSim applies
classifier-free guidance directly on next-token logits:

```text
L_final = L_base + omega * (L_control - L_base)
```

The public CATK configs set `cfg_guidance_mode: manual` with the paper scales:
`0.5` for sketch, `0.7` for latent, and `1.0` for text. Set
`cfg_guidance_mode: off` to recover plain conditional decoding, or use
`cfg_guidance_mode: auto` to choose the paper scale from the active modality.

This CFG option is specific to CATK/SMART token decoding. VBD remains on its
diffusion guidance path.
