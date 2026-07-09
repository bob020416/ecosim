# CATK/SMART Upstream Notice

This directory contains a vendored subset of CATK/SMART used by ECoSim CATK
inference and latent-control utilities. It is not a full mirror of the upstream
CATK repository.

Upstream project:

- **Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models**
- Project page: <https://zhejz.github.io/catk>
- Paper: <https://arxiv.org/abs/2412.05334>

If you use the CATK components, please cite the upstream work:

```bibtex
@inproceedings{zhang2025closed,
  title = {Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models},
  author = {Zhang, Zhejun and Karkus, Peter and Igl, Maximilian and Ding, Wenhao and Chen, Yuxiao and Ivanovic, Boris and Pavone, Marco},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025}
}
```

SMART components are included through CATK. See `src/smart/LICENSE.SMART` and
the repository-level `third_party/NOTICE.md`.

For ECoSim usage, follow the repository-level README and docs instead of the
upstream CATK commands.
