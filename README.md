# VeriSight AI

**Evidence-aware media triage and scam-context assistant.** A triage aid, *not* an authenticity oracle.

VeriSight keeps its evidence channels **separate** and never merges them into one "fake" or "scam" probability:

| Channel | What it is | What it is not |
|---|---|---|
| Synthetic-image signal | CNN (MobileNetV3 / EfficientNet-B0) estimating resemblance to the AI-generated class in its training data, with calibrated-or-raw score, abstain band and Grad-CAM | A deepfake detector, a document/offer verifier, or proof an image is authentic |
| Scam-context review | Optional LLM (or built-in rule-based fallback) flagging warning signs in text *you* supply, each tied to quoted evidence | A fraud decision; it never overrules the classifier |
| Provenance (C2PA) | Planned, **not implemented** (shown as "Not checked") | Absence of metadata is inconclusive, never evidence of fakery |
| Next steps | Practical independent verification checks | A guarantee |

## Verification status (read this first)

This repository was generated and checked in a sandbox **without PyTorch** (the wheel could not be downloaded there). Be precise about what has and has not been run:

| Item | Status |
|---|---|
| `ruff check` / `ruff format` | Passed |
| Config, data manifests + leakage checks, calibration maths, LLM adapter/fallback/schema, database layer, analytics (33 tests) | **Passed** (`pytest`) |
| Streamlit app first render (torch stubbed) | Rendered without exceptions; tabs and disabled Analyse button present |
| Preprocessing, model, training, inference, Grad-CAM, release gate, full `pipeline --demo` (tests exist) | **Written but NOT executed** — those tests skip when torch is absent. Run `pytest` on your machine first; expect to fix small issues |
| Docker build, CI workflow, Neon connection, real LLM call, deployment | **Not run** |
| Model performance | **None.** No model has been trained; no metrics exist. Nothing in this repo claims any |

## Quick start (Windows PowerShell shown)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"            # app + training + test tools
pytest                                        # run this first; all tests should now run
python -m verisight_ai.cli pipeline --config configs/default.yaml --demo   # ~1 min plumbing smoke test
```

Reproducible installs: `requirements/torch.txt` + `requirements/dev.lock.txt` (generated with `uv pip compile`, Python 3.11). `pyproject.toml` is the source of truth; regenerate locks when you change dependencies:
`uv pip compile pyproject.toml --extra dev --python-version 3.11 --no-annotate -o requirements/dev.lock.txt` (then remove the torch/CUDA lines, as the committed files do).

## Real workflow

1. Get CIFAKE manually (see `docs/DATA.md`) into `data/cifake/{train,test}/{REAL,FAKE}`.
2. **One command, data to candidate:**
   ```bash
   python -m verisight_ai.cli pipeline --config configs/default.yaml
   ```
   Runs: environment check → data validation + hashed split manifests + leakage check → head training → optional fine-tuning (kept only if validation improves) → temperature scaling (kept only if validation NLL improves) → threshold chosen on validation → **test evaluated once** → optional external set → robustness suite → CPU latency → packaged candidate bundle with model card. Each run writes a new `training_outputs/run-<timestamp>/`; earlier test evidence is never overwritten.
   - Train on Colab T4: clone the repo, `pip install -e ".[train]"`, set `train.device: cuda` (or `auto`), run the same command, download `training_outputs/run-*/candidate/`. For quick CPU experiments set `data.max_per_class: 5000`.
   - `max_per_class`, epochs, image size etc. are all in `configs/default.yaml`.
3. **Tracking:** `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000` (verify against your MLflow version). Logging is explicit: params, per-epoch metrics with `step`, metrics, plots, manifests, config, candidate bundle.
4. **Release (controlled):** `python -m verisight_ai.cli release --candidate <bundle> [--promote]`; rollback with `--rollback`. Details in `docs/RELEASE.md`.
5. **App:** `streamlit run app.py` (or `python -m verisight_ai.cli app`). Without a promoted model the app starts and reports the model as unavailable.

## Configuration (environment)

See `.env.example`. Everything is optional: no DB URL → no scan logging; no LLM key → rule-based review. Never commit `.env`.

## Layout

```
app.py                    Streamlit UI (upload / saved screenshot / camera, results, dashboard)
configs/default.yaml      pipeline config (validated by pydantic)
src/verisight_ai/
  config.py schemas.py    typed settings and result schemas
  data.py                 validation, manifests, leakage checks
  preprocessing.py        safe decode (size/pixel/format limits) + transforms
  model.py train.py       architectures, staged freezing, explicit MLflow logging
  calibration.py          temperature scaling, threshold, abstain policy
  evaluate.py             metrics, plots, robustness suite, CPU latency
  explainability.py       Grad-CAM (synthetic class)
  inference.py            shared predictor (app, gate, tests); checksum-verified bundles
  release.py              packaging, release gate, promote, rollback
  pipeline.py cli.py      one orchestrator, one CLI
  llm_assistant.py        provider-agnostic adapter + deterministic fallback
  database.py analytics.py  Neon/SQLAlchemy, aggregates only
tests/                    pytest (no GPU, no secrets, no real dataset needed)
docs/                     DATA.md, RELEASE.md, PRIVACY.md
```

## Design decisions worth knowing

- **Test set discipline:** calibration and thresholds use validation data only; the internal test set is evaluated exactly once with the frozen policy.
- **Decision policy:** three outcomes (`synthetic-like`, `no strong signal`, `inconclusive`) via a configurable abstain band. Scores are labelled "raw" unless calibration was fitted *and* accepted.
- **LLM role:** receives structured classifier output + your text (never the image), only after an explicit opt-in checkbox. Output is schema-validated; the assessment is forced to the classifier's; indicators whose quoted evidence isn't in your text are dropped; any failure falls back to the deterministic report. User text is treated as untrusted data.
- **Privacy:** images are never stored; no filenames or message text in DB or logs; dashboard shows aggregates only (see `docs/PRIVACY.md`). Define a retention period before going public.
- **Monitoring:** `logging` + the dashboard tab + `verisight health`. Usage counts are app activity, **not** real-world scam rates; user feedback is opinion, not verified truth.
- **Deployment is manual** (no `deploy.yml` — it could not be tested). See `docs/RELEASE.md`.

## Not included / next

OCR, C2PA verification and video frame sampling are Milestone 5 extensions and intentionally absent. An external evaluation set must be chosen and licence-checked by you; until then the pipeline logs that domain shift was **not** evaluated, and any results table you publish must say so.

## Known limitations

- CIFAKE is 32×32 images from one generator family; results will not transfer to social-media photos, screenshots or deepfake faces. Expect a large drop on unfamiliar data and report it.
- The camera widget captures the webcam, not the desktop screen. Screenshots must be saved and uploaded.
- Release-gate thresholds in `configs/default.yaml` are placeholders chosen by the project, not derived from data; revisit after your first baseline.
