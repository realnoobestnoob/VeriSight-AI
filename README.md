# VeriSight AI

**AI-generated image detector.** Upload or photograph an image to get a probabilistic estimate of whether it resembles AI-generated content from the model's training data.

VeriSight is a **triage aid**, not an authenticity oracle. A high score means resemblance to the AI-generated class seen during training — it is not proof that an image is fake, and a low score is not proof that it is authentic.

---

## Verification status (read this first)

This repository was generated and checked in a sandbox **without PyTorch** (the wheel could not be downloaded there). Be precise about what has and has not been run:

| Item | Status |
|---|---|
| `ruff check` / `ruff format` | ✅ Passed |
| Config, data manifests + leakage checks, calibration maths, LLM adapter/fallback/schema, database layer, analytics (33 tests) | ✅ **Passed** (`pytest`) |
| Streamlit app first render (torch stubbed) | ✅ Rendered without exceptions; upload input and disabled Analyse button present |
| Preprocessing, model, training, inference, Grad-CAM, release gate, full `pipeline --demo` | ⚠️ **Written but NOT executed** — those tests skip when torch is absent. Run `pytest` locally first; expect to fix small issues |
| Docker build, CI workflow, Neon connection, deployment | ❌ **Not run** |
| Model performance | ❌ **None.** No model has been trained; no metrics exist. Nothing in this repo claims any |

---

## Part 1 — Local setup

### Step 1 — Prerequisites

Make sure you have the following installed before starting:

- Python 3.11 (recommended; 3.10+ required)
- Git
- A terminal running **bash** (Linux, macOS, or WSL2 on Windows)

Check your Python version:

```bash
python3 --version
```

---

### Step 2 — Clone the repository

```bash
git clone https://github.com/<your-username>/verisight-ai.git
cd verisight-ai
```

---

### Step 3 — Create and activate an isolated virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your shell prompt. Every command from here on runs inside this environment. To leave it:

```bash
deactivate
```

To re-enter it later:

```bash
source .venv/bin/activate
```

---

### Step 4 — Install PyTorch (CPU)

PyTorch is installed from a separate index to get the smaller CPU-only wheel:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements/torch.txt
```

This downloads about 200 MB. If you plan to train on a GPU (e.g. Google Colab T4), skip this step; the default PyPI wheel on Colab already includes CUDA support.

---

### Step 5 — Install the project and all development dependencies

```bash
pip install -r requirements/dev.lock.txt
pip install --no-deps -e .
```

`-e .` installs the package in editable mode so changes to `src/verisight_ai/` take effect immediately without reinstalling.

---

### Step 6 — Verify the installation

Run the test suite. All 33+ tests should pass; the torch-dependent tests (preprocessing, inference, release gate, end-to-end smoke) require torch and will now run as well:

```bash
pytest
```

Run a quick formatting and lint check:

```bash
ruff format --check .
ruff check .
```

Import-level smoke test (should print nothing and exit 0):

```bash
python -c "import verisight_ai, verisight_ai.cli, verisight_ai.pipeline"
python -c "from verisight_ai.config import load_config; load_config('configs/default.yaml')"
```

---

### Step 7 — Configure secrets (optional)

Copy the example file and fill in any values you want to use. **Never commit `.env`.**

```bash
cp .env.example .env
```

Open `.env` in your editor. Every variable is optional:

- `VERISIGHT_DATABASE_URL` — Neon PostgreSQL connection string; leave empty to disable scan logging.
- `VERISIGHT_LLM_PROVIDER` / `VERISIGHT_LLM_API_KEY` / `VERISIGHT_LLM_MODEL` — reserved for future use; set provider to `none`.

The app works with all values empty.

---

## Part 2 — Getting data

### Step 8 — Download Defactify (primary dataset, recommended)

Defactify contains 96 000 images from MS COCO (real) and five modern AI generators. No account is needed; it is downloaded via the Hugging Face `datasets` library.

**Disk space:** approximately 10 GB after conversion to JPEG files on disk.  
**Time:** 20–60 minutes depending on your connection.

#### Step 8a — Install the datasets library

```bash
pip install datasets tqdm
```

`datasets` is already listed in `requirements/dev.lock.txt`; this step is only needed if you installed with `--no-deps` or skipped the full dev install.

#### Step 8b — Run the download and convert script

```bash
python scripts/download_defactify.py --out data/defactify
```

The script streams each split from Hugging Face and writes images to disk in the REAL/FAKE folder layout the pipeline expects:

```
data/
└── defactify/
    ├── train/
    │   ├── REAL/       (7 000 MS COCO photos)
    │   └── FAKE/       (35 000 AI-generated images across 5 generators)
    ├── val/
    │   ├── REAL/       (1 500)
    │   └── FAKE/       (7 500)
    └── test/
        ├── REAL/       (7 500)
        └── FAKE/       (37 500)
```

> **Quick test:** Add `--max-per-split 1000` to download only 1 000 images per split and verify the pipeline works before committing to the full download.

```bash
python scripts/download_defactify.py --out data/defactify --max-per-split 1000
```

#### Step 8c — Record your download date

Open `configs/defactify.yaml` and fill in today's date:

```yaml
data:
  dataset_version: "2026-<MM>-<DD>"
```

Also update `docs/DATA.md` with your retrieval date as instructed there.

---

### Step 9 — Run the full pipeline

Use `configs/defactify.yaml` for Defactify (recommended) or `configs/default.yaml` for CIFAKE.

**Defactify:**

```bash
python -m verisight_ai.cli pipeline --config configs/defactify.yaml
```

**CIFAKE:**

```bash
python -m verisight_ai.cli pipeline --config configs/default.yaml
```

This single command runs every stage in order:

1. **Environment validation** — checks paths, write permissions, device.
2. **Data validation** — confirms class folders, counts, image formats, and corrupt files; detects cross-split exact duplicates and removes leaks.
3. **Split manifests** — writes reproducible, hashed `manifest.csv` and `manifest_meta.json` to `training_outputs/run-<timestamp>/data/`.
4. **Head training** — trains only the classifier head with the backbone frozen.
5. **Fine-tuning** — unfreezes the last N backbone blocks and trains at a lower learning rate; kept only if validation loss improves over the head stage.
6. **Temperature calibration** — fits a single temperature on validation data; applied only if validation NLL improves.
7. **Decision threshold** — chosen by maximising validation F1; an abstain band produces the "inconclusive" outcome.
8. **Test evaluation** — the internal test set is evaluated exactly once with the frozen policy.
9. **Robustness suite** — tests JPEG recompression (q50, q75), resize round-trip, centre crop, and screenshot-like processing on a subset of test images.
10. **CPU latency benchmark** — single-image forward pass timing.
11. **Candidate packaging** — exports weights, metadata, calibration, thresholds, a model card, and an MLflow run link.

Each run writes to a fresh timestamped folder; earlier test evidence is never overwritten.

**Smoke test without real data (~1 min, validates plumbing only):**

```bash
python -m verisight_ai.cli pipeline --config configs/defactify.yaml --demo
```

**Resume a failed run** (skips completed stages, picks up from the last checkpoint):

```bash
python -m verisight_ai.cli pipeline \
  --config configs/defactify.yaml \
  --resume training_outputs/run-<timestamp>
```

**Colab T4 training:** clone the repo in Colab, run `pip install -e ".[train]"`, mount or upload your `data/defactify/` folder, set `train.device: auto` in the config, then run the same pipeline command. For quick CPU experiments add `max_per_class: 5000` under `data:` in your config.

---

### Step 10 — View experiment tracking in MLflow

In a separate terminal (with the venv active):

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Then open `http://127.0.0.1:5000` in your browser. Every run logs parameters, per-epoch metrics, plots, manifests, and the candidate bundle.

> Verify the exact command against your installed MLflow version (`mlflow --version`) if the server fails to start.

---

### Step 11 — Run the release gate and promote a model

After a training run, check whether the candidate passes the acceptance criteria:

```bash
python -m verisight_ai.cli release \
  --candidate training_outputs/run-<timestamp>/candidate/<release-id>
```

If the gate passes and you are happy with the results, promote the model to active:

```bash
python -m verisight_ai.cli release \
  --candidate training_outputs/run-<timestamp>/candidate/<release-id> \
  --promote
```

This copies the bundle to `model_artifacts/releases/<id>/` and updates `model_artifacts/current.json`. To roll back to the previous release:

```bash
python -m verisight_ai.cli release --rollback
```

> The acceptance thresholds in `release:` in your config are **placeholders**. Update them after your first baseline run to reflect your measured results. Each dataset (Defactify vs CIFAKE) may warrant different thresholds.

---

## Part 4 — Running the app

### Step 12 — Start the web app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

Or use the CLI shortcut:

```bash
python -m verisight_ai.cli app
```

The app starts even if no model has been promoted yet (it reports the model as unavailable). Once a model is promoted (Step 11), the Analyse button becomes active.

---

### Step 13 — Check app health

After starting (or after deployment), run:

```bash
python -m verisight_ai.cli health
```

This checks model loading and database connectivity. Exit code is 0 on success and 1 if anything is unavailable.

---

## Part 5 — Docker

### Step 14 — Build the Docker image

The Docker image is for serving the app with a pre-trained model. It does not run training.

Before building, make sure a model has been promoted (Step 11) so `model_artifacts/current.json` exists:

```bash
docker build -t verisight-ai .
```

### Step 15 — Run the container locally

```bash
docker run --rm -p 8501:8501 \
  -e VERISIGHT_DATABASE_URL="" \
  verisight-ai
```

Then open `http://localhost:8501`. Secrets are always passed at runtime via `-e`; they are never baked into the image.

### Step 16 — Smoke-test the container

```bash
curl -fsS http://localhost:8501/_stcore/health && echo "healthy"
```

---

## Part 6 — CI/CD and deployment

### Step 17 — Push to GitHub and run CI

Push to a branch or open a pull request. GitHub Actions (`.github/workflows/ci.yml`) will:

1. Install Python 3.11 and the pinned dependencies.
2. Run `ruff check` and `ruff format --check`.
3. Run `pytest`.
4. Run the import/config smoke test.
5. Build the Docker image and probe the health endpoint.

CI **does not** run training and **does not** need secrets or GPU access.

### Step 18 — Deploy (manual)

Deployment is intentionally manual because it could not be tested in the generation environment. A tested CI run (Step 17) must pass before deploying.

Suggested route — Render Docker web service:

1. Create a new Web Service on <https://render.com> and point it at your GitHub repository.
2. Set the environment to "Docker".
3. Add all secrets from `.env.example` under Environment in the Render dashboard.
4. Trigger a deploy. After it goes live, run:

```bash
curl -fsS https://<your-render-url>/_stcore/health
```

If you use Streamlit Community Cloud instead, set secrets in the Streamlit dashboard and set the main file to `app.py`. Note that Community Cloud cannot serve a Docker image directly; it runs `streamlit run app.py`.

---

## Regenerating dependency lock files

Run these when `pyproject.toml` changes (requires `uv`):

```bash
pip install uv

# Runtime lock (no torch/CUDA — those stay in requirements/torch.txt)
uv pip compile pyproject.toml --python-version 3.11 --no-annotate \
  -o requirements/runtime.lock.txt

# Development lock (includes training + test tools)
uv pip compile pyproject.toml --extra dev --python-version 3.11 --no-annotate \
  -o requirements/dev.lock.txt

# Remove torch, torchvision, triton and nvidia-* lines from both files
# (they are managed by requirements/torch.txt)
```

---

## Repository layout

```
app.py                    Streamlit UI (upload / camera, result, Grad-CAM, feedback)
configs/
  default.yaml            Pipeline config for CIFAKE
  defactify.yaml          Pipeline config for Defactify (primary)
scripts/
  download_defactify.py   Downloads and converts Defactify from Hugging Face to disk
src/verisight_ai/
  config.py               Typed pipeline and app settings
  schemas.py              Typed result schemas
  data.py                 Data validation, split manifests, leakage checks
  preprocessing.py        Safe image decode (format/size/pixel limits) + transforms
  model.py                Architectures, staged freezing, device selection
  train.py                Training loop with explicit MLflow logging
  calibration.py          Temperature scaling, threshold selection, abstain policy
  evaluate.py             Metrics, plots, robustness suite, CPU latency benchmark
  explainability.py       Grad-CAM for the synthetic class
  inference.py            Shared predictor used by the app, the gate and tests
  release.py              Candidate packaging, release gate, promote, rollback
  pipeline.py             Single orchestrator: all stages in dependency order
  cli.py                  One entry point: pipeline / release / health / app / demo-data
  database.py             Neon / SQLAlchemy; privacy-safe aggregates only
  analytics.py            Usage aggregations (available for custom dashboards)
  tracking.py             Thin MLflow wrapper (explicit, no autologging)
  demo_data.py            Tiny synthetic smoke dataset (proves nothing about detection)
  logging_setup.py        Standard-library logging; never logs image bytes or secrets
tests/
  conftest.py             Shared fixtures (tiny random model bundle, image helpers)
  test_config.py          Config validation and AppSettings
  test_preprocessing.py   Decode, format, size and pixel-limit rejection
  test_data.py            Manifests, leakage, corrupt files, reproducibility
  test_calibration.py     Temperature scaling, threshold, abstain logic
  test_inference.py       Predictor, checksum guard, explain output
  test_release.py         Gate checks, promote, rollback
  test_llm_schema.py      LLM adapter, fallback, schema enforcement, opt-in gate
  test_database.py        Record, feedback, SQL injection safety, unavailable DB
  test_pipeline_smoke.py  Full demo pipeline smoke (slow; requires torch + sklearn)
docs/
  DATA.md                 How to get both datasets; download instructions; external dataset checklist
  RELEASE.md              Promote, rollback, post-deploy smoke, monitoring
  PRIVACY.md              What is and is not stored; retention policy guidance
requirements/
  torch.txt               CPU PyTorch pin (installed via the whl/cpu index)
  runtime.lock.txt        Pinned app dependencies (no torch/CUDA)
  dev.lock.txt            Pinned dev + train dependencies (no torch/CUDA)
```

---

## Design decisions

- **Test set discipline:** calibration and thresholds use validation data only; the test set is evaluated exactly once with the frozen policy. Each run writes to a fresh folder so earlier test evidence is never overwritten.
- **Decision policy:** three outcomes (`synthetic-like`, `no strong signal`, `inconclusive`) via a configurable abstain band around the threshold. Scores are labelled "raw" unless calibration was fitted *and* accepted.
- **Privacy:** images are not stored; no filenames or message text in DB or logs. Define a retention period before going public (see `docs/PRIVACY.md`).
- **Pipeline resume:** if a training run fails mid-way, re-run with `--resume training_outputs/run-<timestamp>` to skip completed stages and continue from the last checkpoint.
- **Monitoring:** `logging` + `verisight health`. Scan counts in the database are app activity, not real-world AI-image rates; user feedback is opinion, not verified truth.

---

## Known limitations

- **Training data:** real images are from MS COCO (natural photos, reasonable resolution). Results will not transfer reliably to arbitrary social-media images, face swaps, or document forgeries.
- No external evaluation set is included. Until you add one, the pipeline logs that domain shift was **not** evaluated. Any published results table must say so.
- The camera widget captures the webcam, not the desktop screen. Screenshots must be saved as files and uploaded.
- Release-gate metric thresholds in the configs are placeholders; revise them after your first baseline run.
- OCR, C2PA provenance validation, and video frame sampling are planned extensions and are not implemented.