# Release Checklist

Before making the GitHub repository public:

- Choose and add the final repository license.
- Confirm redistribution terms for vendored VBD, CATK, and SMART files.
- Add checkpoint download links to `docs/checkpoints.md`.
- Add data preparation/download links to `docs/data.md`.
- Run at least one real scenario for each released backend:
  - CATK GT sketch
  - CATK GT latent
  - CATK GT text
- Run BehaviorVAE latent export on a small VBD subset and verify both VBD and
  CATK latent payload formats.
- Add paper/project links after they are public.
- Add teaser and method figures under `assets/` if they can be redistributed.
