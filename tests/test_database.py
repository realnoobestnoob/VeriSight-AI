from verisight_ai.analytics import count_by, operational_summary, scans_per_day
from verisight_ai.database import NullRepository, SqlRepository, make_repository, normalise_url


def _scan(repo, **kw):
    base = dict(
        input_type="upload",
        file_type="png",
        status="succeeded",
        llm_used=False,
        inference_latency_ms=12.0,
        model_release_id="r1",
        decision_status="inconclusive",
    )
    base.update(kw)
    return repo.record_scan(**base)


def test_record_and_read(tmp_path):
    repo = SqlRepository(f"sqlite:///{tmp_path / 't.db'}")
    sid = _scan(repo)
    _scan(repo, status="failed", error_category="model_unavailable", inference_latency_ms=None)
    assert sid and repo.ping()
    df = repo.scan_frame()
    s = operational_summary(df)
    assert s["total"] == 2 and s["failure_rate"] == 0.5
    assert len(scans_per_day(df)) == 1
    assert count_by(df, "input_type").iloc[0]["count"] == 2


def test_feedback_validation(tmp_path):
    repo = SqlRepository(f"sqlite:///{tmp_path / 't.db'}")
    sid = _scan(repo)
    assert repo.record_feedback(sid, "helpful")
    assert not repo.record_feedback(sid, "'; DROP TABLE scans;--")
    assert repo.feedback_frame().iloc[0]["feedback_label"] == "helpful"


def test_sql_injection_in_text_field_is_inert(tmp_path):
    repo = SqlRepository(f"sqlite:///{tmp_path / 't.db'}")
    _scan(repo, model_release_id="x'); DROP TABLE scans;--")
    assert len(repo.scan_frame()) == 1


def test_unavailable_db_fails_soft(tmp_path):
    repo = SqlRepository(f"sqlite:///{tmp_path / 'missing_dir' / 'x.db'}", create=False)
    assert repo.record_scan(input_type="upload", status="failed") is None
    assert not repo.ping()
    assert repo.scan_frame().empty


def test_no_url_gives_null_repo():
    repo = make_repository(None)
    assert isinstance(repo, NullRepository) and repo.record_scan() is None


def test_bad_url_gives_null_repo():
    assert isinstance(make_repository("not a url"), NullRepository)


def test_neon_url_normalised():
    assert normalise_url("postgresql://u:p@h/db?sslmode=require").startswith(
        "postgresql+psycopg2://"
    )
    assert normalise_url("postgres://u:p@h/db").startswith("postgresql+psycopg2://")
