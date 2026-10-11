# Data

VeriSight never downloads data automatically.

## CIFAKE (baseline)
1. Download manually from Kaggle: <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images> (needs a free Kaggle account; check the dataset's licence/terms yourself).
2. Extract so the layout is:
   ```
   data/cifake/train/REAL/*.jpg   data/cifake/train/FAKE/*.jpg
   data/cifake/test/REAL/*.jpg    data/cifake/test/FAKE/*.jpg
   ```
3. Record your retrieval date here: `YYYY-MM-DD` (and update `data.dataset_version` in `configs/default.yaml`).

The pipeline keeps the official `test/` folder as the internal test set and carves validation out of `train/` (stratified, seeded). Exact duplicate files that appear across splits are removed from train/val.

**Limits:** CIFAKE images are 32x32 and come from one generator family. Results say nothing reliable about deepfake faces, scam posters, screenshots or arbitrary social-media images.

## External evaluation set (not included)
Set `data.external_root` (same `REAL/` + `FAKE/` layout) and `data.external_name`. Candidates such as WildFake must have their access, licence and labels verified first. If none is used, the pipeline logs that domain shift was **not** evaluated, and the README results table must say so.

## Smoke-test data
`python -m verisight_ai.cli demo-data` / `pipeline --demo` generate a tiny **synthetic** dataset (stripe patterns). It exists only to test the plumbing and proves nothing about detection.
