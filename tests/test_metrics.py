"""Forecast-metric tests."""

import pytest

from footy.eval.metrics import rps_1x2, log_loss_1x2, brier_1x2, summarize


def test_rps_perfect_is_zero():
    assert rps_1x2((1.0, 0.0, 0.0), "home") == pytest.approx(0.0)


def test_rps_worst_case():
    # certain home, but away happened: cumulative errors (1,0)->(1,0) vs (0,0,1)
    assert rps_1x2((1.0, 0.0, 0.0), "away") == pytest.approx(1.0)


def test_rps_ordinal_draw_less_wrong_than_away():
    # predicting home; draw happening should score better than away happening
    p = (0.6, 0.3, 0.1)
    assert rps_1x2(p, "draw") < rps_1x2(p, "away")


def test_log_loss_and_brier():
    assert log_loss_1x2((0.5, 0.3, 0.2), "home") == pytest.approx(-__import__("math").log(0.5))
    assert brier_1x2((1.0, 0.0, 0.0), "home") == pytest.approx(0.0)


def test_summarize():
    fc = [((0.6, 0.3, 0.1), "home"), ((0.2, 0.3, 0.5), "away")]
    s = summarize(fc)
    assert s["n"] == 2 and s["accuracy"] == 1.0
    assert 0 <= s["rps"] <= 1
