import pytest

from verisight_ai.config import AppSettings, load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.architecture == "mobilenet_v3_small"
    assert 0 < cfg.data.val_fraction < 0.5


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  architecure: typo\n")
    with pytest.raises(ValueError, match="Invalid config"):
        load_config(p)


def test_out_of_range_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  val_fraction: 0.9\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_missing_file():
    with pytest.raises(ValueError, match="not found"):
        load_config("nope.yaml")


def test_app_settings_from_env():
    s = AppSettings.from_env({"VERISIGHT_MAX_UPLOAD_MB": "2", "VERISIGHT_LLM_PROVIDER": "none"})
    assert s.max_upload_bytes == 2 * 1024 * 1024
    assert s.database_url is None


def test_app_settings_bad_value():
    with pytest.raises(ValueError):
        AppSettings.from_env({"VERISIGHT_MAX_PIXELS": "abc"})
