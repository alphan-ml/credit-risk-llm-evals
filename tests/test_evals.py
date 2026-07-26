"""Tests: the gate, the metrics, the determinism, the data contract."""
import numpy as np
import pandas as pd
import pytest

from credit_evals.data import MODEL_FEATURES, SLICE_ONLY, load, render_application
from credit_evals.evals import (eval_calibration, eval_cost,
                                eval_discrimination)
from credit_evals.llm_ranker import judge, verify_citations
from credit_evals.ml_model import make_split, predict_pd, train


@pytest.fixture(scope="module")
def df():
    return load()


@pytest.fixture(scope="module")
def fitted(df):
    split = make_split(df)
    model = train(split)
    return split, model


# ── data contract ───────────────────────────────────────────────────
def test_slice_columns_never_reach_the_model(df):
    assert not set(SLICE_ONLY) & set(MODEL_FEATURES)
    app = render_application(df.iloc[0])
    for col in SLICE_ONLY:
        assert col not in app


def test_label_direction(df):
    # 300 of 1000 are defaults in Statlog; if this flips, everything lies.
    assert df["default"].sum() == 300


# ── fabrication gate ────────────────────────────────────────────────
def test_gate_rejects_fabricated_number():
    app = "Loan application:\n- duration: 24\n- amount: 3000"
    assert verify_citations("risky: duration 24 months", app)
    assert not verify_citations("risky: income only 950 DM", app)  # 950 nowhere


def test_gate_fires_end_to_end(df, monkeypatch):
    import credit_evals.llm_ranker as lr
    monkeypatch.setattr(lr, "_stub",
                        lambda app: ("E", "cited a made-up 99999 figure"))
    j = lr.judge(render_application(df.iloc[0]))
    assert not j.grounded and "withheld" in j.rationale


def test_stub_is_deterministic(df):
    app = render_application(df.iloc[7])
    assert judge(app) == judge(app)


# ── metric sanity ───────────────────────────────────────────────────
def test_model_actually_discriminates(fitted):
    split, model = fitted
    d = eval_discrimination(split.y_test, predict_pd(model, split.X_test))
    assert d["auc"] > 0.70, "baseline should beat coin flips comfortably"


def test_calibration_bounded(fitted):
    split, model = fitted
    c = eval_calibration(split.y_test, predict_pd(model, split.X_test))
    assert 0 < c["brier"] < 0.25 and 0 <= c["ece"] < 0.15


def test_cost_uses_official_matrix_and_beats_naive(fitted):
    split, model = fitted
    c = eval_cost(split.y_test, predict_pd(model, split.X_test))
    assert c["cost_matrix"] == {"defaulter_approved": 5, "good_declined": 1}
    assert c["best"]["total_cost"] < c["cost_approve_or_decline_all"]


def test_perfect_and_random_scores_bracket_reality():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    assert eval_discrimination(y, y.astype(float))["auc"] == 1.0
    assert abs(eval_discrimination(y, rng.random(400))["auc"] - 0.5) < 0.1
