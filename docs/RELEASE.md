# Release, deployment, monitoring and rollback

## Promote a model (controlled step)
```bash
python -m verisight_ai.cli pipeline --config configs/default.yaml      # produces a candidate
python -m verisight_ai.cli release --candidate training_outputs/run-*/candidate/<id>            # gate only
python -m verisight_ai.cli release --candidate training_outputs/run-*/candidate/<id> --promote  # gate + promote
```
The gate checks: checksum, model card, MLflow run id + git commit recorded, smoke inference, and the metric thresholds in `release:` (project-defined in `configs/default.yaml`; revisit after your baseline). Promotion copies the bundle to `model_artifacts/releases/<id>/`, updates `model_artifacts/current.json` (with the previous release id) and appends to `release_history.jsonl`.

## Roll back
```bash
python -m verisight_ai.cli release --rollback
```
Points `current.json` back at the previous release. Redeploy afterwards.

## Shipping the model to the host
The hosted app must not depend on local MLflow files. Commit/ship the small promoted bundle (`model_artifacts/current.json` + `releases/<id>/`, MobileNetV3-small is a few MB) with the deployment (remove those paths from `.gitignore` on the release branch, or use Git LFS / a release asset). Verify the checksum in `metadata.json`. Never train at app start.

## Deployment (honest status)
- CI (`.github/workflows/ci.yml`) lints, tests, builds the Docker image and probes the container health endpoint. **It has not been run in the environment where this repo was generated; run it on your first push.**
- **Deployment is MANUAL.** No `deploy.yml` is provided, because deploy hooks differ per host and could not be tested here. Suggested route: Render (Docker web service) or Streamlit Community Cloud. If the host auto-deploys on push, that is *not* gated on CI unless you configure it to wait for checks. Document what you actually configure.
- Secrets (`VERISIGHT_DATABASE_URL`, `VERISIGHT_LLM_API_KEY`) go in the host's secret settings only.

## Post-deploy smoke check
```bash
python -m verisight_ai.cli health     # config, model load, DB ping (exit code 1 on failure)
curl -fsS https://<host>/_stcore/health
```

## Monitoring (lightweight)
- App logs use `logging` and never contain image bytes, filenames, message text, keys or DB URLs.
- Dashboard tab: scan volume, failure rate, p50/p95 latency, model versions, error categories (aggregates only).
- Manual routine: check host logs for `status=failed`; compare failure rate and latency with your measured baseline before setting alert thresholds. Score-distribution drift without verified labels does not prove accuracy changed.
- Promote a new model only after the gate passes **and** you compared its saved evaluation report with the previous release.
