import numpy as np

from verisight_ai.calibration import choose_threshold, decide, fit_temperature, softmax_p_synthetic
from verisight_ai.schemas import DecisionStatus


def test_decide_three_way():
    assert decide(0.9, 0.5, 0.1) is DecisionStatus.SYNTHETIC_LIKE
    assert decide(0.1, 0.5, 0.1) is DecisionStatus.NO_STRONG_SIGNAL
    assert decide(0.55, 0.5, 0.1) is DecisionStatus.INCONCLUSIVE


def test_temperature_scaling_improves_overconfident_logits():
    rng = np.random.default_rng(0)
    n = 2000
    labels = rng.integers(0, 2, n)
    signal = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 1.5, n)  # noisy
    logits = np.stack([-signal * 8, signal * 8], axis=1)  # far too confident
    out = fit_temperature(logits, labels)
    assert out["applied"] and out["temperature"] > 1.5
    assert out["val_nll_after"] < out["val_nll_before"]


def test_softmax_in_unit_interval():
    p = softmax_p_synthetic(np.array([[0.0, 100.0], [100.0, 0.0]]))
    assert np.all((p >= 0) & (p <= 1)) and p[0] > 0.99 and p[1] < 0.01


def test_threshold_selection_separable():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t = choose_threshold(scores, labels)
    assert 0.3 < t["threshold"] <= 0.7 and t["f1"] == 1.0
