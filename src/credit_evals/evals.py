"""The eval harness — the actual subject of this repo.

Seven evals, each answering one plain question about the system:

1. discrimination  — can the ML path order risk at all? (AUC, KS)
2. calibration     — when it says 30% PD, does ~30% default? (Brier, ECE,
                     reliability table)
3. cost            — using the dataset's OWN 5:1 cost matrix, where is the
                     decision threshold, and what does it cost? (sweep)
4. agreement       — do the two paths rank risk the same way? (Spearman,
                     Kendall, and where they disagree hardest)
5. slices          — same metrics on groups the model never saw as
                     features: age bands, sex/status, foreign worker
6. groundedness    — how often does the LLM path cite figures that are
                     not in the application? (gate fire rate)
7. abstention      — does the LLM path know when it can't judge?

Every eval returns a plain dict; the report renderer turns them into one
markdown file. No dashboard, no framework — evals as code you can read.
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import SLICE_ONLY, TARGET, render_application
from .llm_ranker import GRADE_TO_RANK, judge

COST_FN = 5  # defaulter approved  (dataset's official cost matrix)
COST_FP = 1  # good applicant declined


# ── 1 · discrimination ──────────────────────────────────────────────
def eval_discrimination(y, pd_scores) -> dict:
    order = np.argsort(pd_scores)
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / y_sorted.sum()
    cum_good = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    return {"auc": round(float(roc_auc_score(y, pd_scores)), 4),
            "ks": round(float(np.max(np.abs(cum_bad - cum_good))), 4),
            "n": int(len(y)), "base_default_rate": round(float(y.mean()), 4)}


# ── 2 · calibration ─────────────────────────────────────────────────
def eval_calibration(y, pd_scores, bins: int = 10) -> dict:
    df = pd.DataFrame({"y": y, "p": pd_scores})
    df["bin"] = np.minimum((df.p * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for b, g in df.groupby("bin"):
        rows.append({"bin": f"{b/bins:.1f}–{(b+1)/bins:.1f}",
                     "n": int(len(g)),
                     "mean_predicted": round(float(g.p.mean()), 3),
                     "observed_default": round(float(g.y.mean()), 3)})
        ece += (len(g) / len(df)) * abs(g.p.mean() - g.y.mean())
    return {"brier": round(float(brier_score_loss(y, pd_scores)), 4),
            "ece": round(float(ece), 4), "reliability": rows}


# ── 3 · cost curve on the dataset's own matrix ──────────────────────
def eval_cost(y, pd_scores) -> dict:
    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        approve = pd_scores < t
        fn = int(((y == 1) & approve).sum())     # defaulters approved
        fp = int(((y == 0) & ~approve).sum())    # good customers declined
        sweep.append({"threshold": float(t), "defaulters_approved": fn,
                      "good_declined": fp,
                      "total_cost": COST_FN * fn + COST_FP * fp})
    best = min(sweep, key=lambda r: r["total_cost"])
    naive = min(COST_FN * int(y.sum()),          # approve everyone
                COST_FP * int((y == 0).sum()))   # decline everyone
    return {"cost_matrix": {"defaulter_approved": COST_FN,
                            "good_declined": COST_FP},
            "best": best, "cost_approve_or_decline_all": naive,
            "sweep": sweep}


# ── 4-7 · the LLM-path evals share one pass over the eval frame ─────
def run_llm_path(df_test: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, row in df_test.iterrows():
        app = render_application(row)
        j = judge(app)
        out.append({"grade": j.grade, "grounded": j.grounded,
                    "source": j.source,
                    "rank": GRADE_TO_RANK.get(j.grade, np.nan)})
    return pd.DataFrame(out, index=df_test.index)


def eval_agreement(pd_scores, llm: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    m = llm["rank"].notna()
    rho, _ = spearmanr(pd_scores[m], llm.loc[m, "rank"])
    tau, _ = kendalltau(pd_scores[m], llm.loc[m, "rank"])
    d = pd.DataFrame({"pd": pd_scores, "rank": llm["rank"],
                      "y": df_test[TARGET].to_numpy()})[m]
    # sharpest disagreements: model says safe, stub says E — and vice versa
    d["pd_pct"] = d["pd"].rank(pct=True)
    ml_safe_llm_risky = d[(d.pd_pct < 0.3) & (d["rank"] == 4)]
    ml_risky_llm_safe = d[(d.pd_pct > 0.7) & (d["rank"] == 0)]
    def _side(x):
        return {"n": int(len(x)),
                "actual_default_rate": round(float(x.y.mean()), 3) if len(x) else None}
    return {"n_graded": int(m.sum()),
            "spearman_rho": round(float(rho), 4),
            "kendall_tau": round(float(tau), 4),
            "ml_safe_llm_risky": _side(ml_safe_llm_risky),
            "ml_risky_llm_safe": _side(ml_risky_llm_safe)}


def eval_slices(df_test: pd.DataFrame, y, pd_scores, threshold: float) -> dict:
    df = df_test.copy()
    df["_pd"], df["_y"] = pd_scores, y
    df["_approve"] = pd_scores < threshold
    df["age_band"] = pd.cut(df["age"], [18, 25, 35, 50, 100],
                            labels=["19-25", "26-35", "36-50", "51+"])
    out = {}
    for col in ["age_band", "personal_status_sex", "foreign_worker"]:
        rows = []
        for val, g in df.groupby(col, observed=True):
            if len(g) < 15:
                continue
            r = {"group": str(val), "n": int(len(g)),
                 "default_rate": round(float(g._y.mean()), 3),
                 "approval_rate": round(float(g._approve.mean()), 3)}
            if g._y.nunique() == 2:
                r["auc"] = round(float(roc_auc_score(g._y, g._pd)), 3)
            rows.append(r)
        out[col] = rows
    return out


def eval_groundedness(llm: pd.DataFrame) -> dict:
    return {"n": int(len(llm)),
            "gate_fired": int((~llm.grounded).sum()),
            "gate_fire_rate": round(float((~llm.grounded).mean()), 4),
            "source": llm.source.mode()[0]}


def eval_abstention(llm: pd.DataFrame, y) -> dict:
    abst = llm["rank"].isna()
    return {"abstained": int(abst.sum()),
            "abstention_rate": round(float(abst.mean()), 4),
            "default_rate_when_abstained":
                round(float(y[abst].mean()), 3) if abst.any() else None}
