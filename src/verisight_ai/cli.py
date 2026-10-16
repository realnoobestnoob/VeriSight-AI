"""Single entry point: `python -m verisight_ai.cli <command>`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import AppSettings, load_config
from .logging_setup import setup_logging


def _cmd_pipeline(args) -> int:
    from .pipeline import run_pipeline

    cfg = load_config(args.config)

    resume_dir = Path(args.resume) if getattr(args, "resume", None) else None

    if args.demo and resume_dir:
        print("error: --demo and --resume are mutually exclusive", file=sys.stderr)
        return 2

    if args.demo:  # tiny synthetic smoke run, overrides heavy settings
        from .demo_data import make_demo_dataset

        make_demo_dataset(Path(cfg.output_dir) / "demo_data")
        cfg.data.root = Path(cfg.output_dir) / "demo_data"
        cfg.model.pretrained = False
        cfg.model.image_size = 64
        cfg.train.head.epochs, cfg.train.finetune.epochs = 1, 1
        cfg.train.batch_size = 16
        cfg.evaluation.latency_runs, cfg.evaluation.robustness_max_images = 10, 30
        cfg.mlflow.tracking_uri = (
            f"sqlite:///{(Path(cfg.output_dir) / 'mlflow_demo.db').as_posix()}"
        )
        cfg.mlflow.experiment_name = "VeriSight AI (demo smoke)"

    summary = run_pipeline(cfg, resume_dir=resume_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_release(args) -> int:
    from .release import promote, rollback, run_release_gate

    cfg = load_config(args.config)
    if args.rollback:
        print(json.dumps(rollback(args.artifacts_dir), indent=2))
        return 0
    if not args.candidate:
        print("error: --candidate is required (or use --rollback)", file=sys.stderr)
        return 2
    gate = run_release_gate(Path(args.candidate), cfg.release)
    print(json.dumps(gate, indent=2))
    if not gate["passed"]:
        return 1
    if args.promote:
        print(json.dumps(promote(Path(args.candidate), cfg.release, args.artifacts_dir), indent=2))
    else:
        print("Gate passed. Re-run with --promote to make this the active release.")
    return 0


def _cmd_health(args) -> int:
    """Post-deploy / container smoke check: config, model load, optional DB connectivity."""
    from .database import make_repository
    from .inference import ModelUnavailableError, Predictor, resolve_bundle_dir

    s = AppSettings.from_env()
    ok = True
    try:
        pred = Predictor(resolve_bundle_dir(s.model_dir, s.artifacts_dir))
        print(f"model: OK ({pred.release_id})")
    except ModelUnavailableError as exc:
        print(f"model: UNAVAILABLE ({exc})")
        ok = False
    repo = make_repository(s.database_url)
    if s.database_url:
        db_ok = repo.ping()
        print(f"database: {'OK' if db_ok else 'UNAVAILABLE'}")
        ok = ok and db_ok
    else:
        print("database: not configured (scan logging disabled)")
    print(f"llm: {s.llm_provider}")
    return 0 if ok else 1


def _cmd_app(args) -> int:
    app = Path(__file__).resolve().parents[2] / "app.py"
    if not app.is_file():
        app = Path("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def _cmd_demo_data(args) -> int:
    from .demo_data import make_demo_dataset

    print(make_demo_dataset(args.out, n_per_class=args.n))
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(prog="verisight", description="VeriSight AI command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pipeline", help="Run data validation -> train -> evaluate -> package")
    sp.add_argument("--config", default="configs/default.yaml")
    sp.add_argument(
        "--demo",
        action="store_true",
        help="Run on a tiny synthetic dataset (smoke test only; proves nothing)",
    )
    sp.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help=(
            "Resume a failed run from its last checkpoint. "
            "Pass the failed run's directory, e.g. "
            "training_outputs/run-20261013-154652. "
            "Completed stages are skipped; their outputs are loaded from disk."
        ),
    )
    sp.set_defaults(fn=_cmd_pipeline)

    sr = sub.add_parser("release", help="Run the release gate; optionally promote or roll back")
    sr.add_argument("--config", default="configs/default.yaml")
    sr.add_argument("--candidate")
    sr.add_argument("--promote", action="store_true")
    sr.add_argument("--rollback", action="store_true")
    sr.add_argument("--artifacts-dir", default="model_artifacts")
    sr.set_defaults(fn=_cmd_release)

    sh = sub.add_parser("health", help="Check config, model loading and database connectivity")
    sh.set_defaults(fn=_cmd_health)
    sa = sub.add_parser("app", help="Start the Streamlit app")
    sa.set_defaults(fn=_cmd_app)
    sd = sub.add_parser("demo-data", help="Create a tiny synthetic smoke dataset")
    sd.add_argument("--out", default="data/demo")
    sd.add_argument("--n", type=int, default=60)
    sd.set_defaults(fn=_cmd_demo_data)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
