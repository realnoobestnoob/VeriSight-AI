"""Application database (Neon PostgreSQL in production, SQLite in tests).

Privacy: no uploaded media, filenames or message text are stored. All SQL is parameterised
through SQLAlchemy Core. Every operation fails soft: the app keeps working if the DB is down.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Protocol

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

log = logging.getLogger(__name__)
metadata = MetaData()

scans = Table(
    "scans",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("input_type", String(32), nullable=False),
    Column("file_type", String(16)),
    Column("file_hash", String(64)),
    Column("predicted_class", String(8)),
    Column("raw_score", Float),
    Column("calibrated_score", Float),
    Column("decision_status", String(24)),
    Column("model_release_id", String(96)),
    Column("training_run_id", String(64)),
    Column("inference_latency_ms", Float),
    Column("status", String(20), nullable=False),
    Column("error_category", String(32)),
    Column("llm_used", Boolean, nullable=False, default=False),
    Column("provenance_status", String(16)),
    Column("video_frames_analysed", Integer),
)

scan_feedback = Table(
    "scan_feedback",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("scan_id", String(36), ForeignKey("scans.id"), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("feedback_label", String(16), nullable=False),
    Column("verified_label", String(8)),  # only set via an independent verification process
)

FEEDBACK_LABELS = {"helpful", "not_helpful", "unsure"}


class Repository(Protocol):
    enabled: bool

    def ping(self) -> bool: ...
    def record_scan(self, **kwargs) -> str | None: ...
    def record_feedback(self, scan_id: str, label: str) -> bool: ...
    def scan_frame(self, limit: int = 50000) -> pd.DataFrame: ...
    def feedback_frame(self, limit: int = 50000) -> pd.DataFrame: ...


class NullRepository:
    """Used when no DATABASE_URL is configured: scan logging is simply off."""

    enabled = False

    def ping(self) -> bool:
        return False

    def record_scan(self, **kwargs) -> str | None:
        return None

    def record_feedback(self, scan_id: str, label: str) -> bool:
        return False

    def scan_frame(self, limit: int = 50000) -> pd.DataFrame:
        return pd.DataFrame()

    def feedback_frame(self, limit: int = 50000) -> pd.DataFrame:
        return pd.DataFrame()


def normalise_url(url: str) -> str:
    """Neon gives postgresql://...; SQLAlchemy needs an explicit driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]
    return url


class SqlRepository:
    enabled = True

    def __init__(self, url: str, create: bool = True):
        self.engine: Engine = create_engine(
            normalise_url(url), pool_pre_ping=True, pool_recycle=300
        )
        if create:
            try:
                metadata.create_all(self.engine)
            except SQLAlchemyError:
                log.error("Database schema creation failed (database_unavailable).")

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def record_scan(self, **kw) -> str | None:
        scan_id = str(uuid.uuid4())
        row = {c.name: kw.get(c.name) for c in scans.columns}
        row.update(
            id=scan_id,
            created_at=dt.datetime.now(dt.timezone.utc),
            llm_used=bool(kw.get("llm_used", False)),
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(scans), row)
            return scan_id
        except SQLAlchemyError:
            log.error("Failed to record scan (database_unavailable).")  # no payload in the log
            return None

    def record_feedback(self, scan_id: str, label: str) -> bool:
        if label not in FEEDBACK_LABELS:
            return False
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(scan_feedback),
                    {
                        "id": str(uuid.uuid4()),
                        "scan_id": scan_id,
                        "created_at": dt.datetime.now(dt.timezone.utc),
                        "feedback_label": label,
                        "verified_label": None,
                    },
                )
            return True
        except SQLAlchemyError:
            log.error("Failed to record feedback (database_unavailable).")
            return False

    def scan_frame(self, limit: int = 50000) -> pd.DataFrame:
        cols = [
            scans.c.created_at,
            scans.c.input_type,
            scans.c.status,
            scans.c.error_category,
            scans.c.inference_latency_ms,
            scans.c.model_release_id,
            scans.c.decision_status,
            scans.c.llm_used,
            scans.c.raw_score,
        ]
        try:
            with self.engine.connect() as conn:
                q = select(*cols).order_by(scans.c.created_at.desc()).limit(limit)
                return pd.DataFrame(conn.execute(q).fetchall(), columns=[c.name for c in cols])
        except SQLAlchemyError:
            log.error("Failed to read scan aggregates (database_unavailable).")
            return pd.DataFrame()

    def feedback_frame(self, limit: int = 50000) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                q = select(scan_feedback.c.feedback_label, func.count()).group_by(
                    scan_feedback.c.feedback_label
                )
                return pd.DataFrame(conn.execute(q).fetchall(), columns=["feedback_label", "count"])
        except SQLAlchemyError:
            return pd.DataFrame()


def make_repository(url: str | None) -> Repository:
    if not url:
        return NullRepository()
    try:
        return SqlRepository(url)
    except Exception:  # noqa: BLE001 - bad URL / missing driver must not crash the app
        log.error("Could not initialise database repository (database_unavailable).")
        return NullRepository()
