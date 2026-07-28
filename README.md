# credit-risk-llm-evals

An evaluation harness for a dual-path credit risk system — an ML classifier
and an LLM ranker scored side by side on **real loan outcomes**. The model
and ranker are deliberately modest; the harness is the product. The question
it answers: *before you trust a scoring system, what should you have measured?*

This repo also closes a loop: my first public data project (2019) was an
RMarkdown EDA on Lending Club loans. This is that domain revisited with what
I now know matters — outcomes, calibration, costs, slices, and gates, not
just fit statistics.

## Architecture

```
                    ┌── ML path: GBM ──────────► PD score ─┐
application row ────┤                                      ├──► eval harness ──► evals/report.md
                    └── LLM path: grade A–E ──► rank ──────┘
                         │ rationale must cite only values
                         ▼ present in the application
                    groundedness gate (verify_citations)
```

Seven evals, each one plain question (`src/credit_evals/evals.py`):
**discrimination** (AUC/KS) · **calibration** (Brier/ECE/reliability) ·
**cost** on the dataset's own 5:1 matrix · **ML↔LLM agreement**
(Spearman/Kendall + hardest disagreements) · **slices** on attributes the
model never saw · **groundedness** (gate fire rate) · **abstention**.

## Run it (no keys needed)

```bash
git clone <this repo> && cd credit-risk-llm-evals
pip install -r requirements.txt
cd src && python -m credit_evals.run_evals
```

Tests: `pip install -r requirements-dev.txt && cd src && python -m pytest ../tests`

## Real results (this exact code, this exact data)

```
auc 0.7868 · ks 0.454 · brier 0.1612 · ece 0.0548
best_threshold 0.20 → total cost 147  (vs 210 for decline-everyone)
spearman(ML, LLM-stub) 0.1679 · gate_fire_rate 0.0
```

Three findings the harness surfaced, none visible from AUC alone:

1. **The cost eval moves the threshold to 0.20.** With the dataset's official
   asymmetric costs (a missed defaulter = 5× a lost customer), the naive 0.5
   cutoff is badly wrong; the sweep lands at 0.20 and beats decline-everyone
   by 30%.
2. **The slice eval caught an age cliff.** Overall AUC 0.79 hides AUC 0.657
   for applicants 19–25 vs 0.932 for 51+ — the model barely ranks young
   applicants. Age is not a model feature; the harness checks slices
   precisely because exclusion from features is not exclusion from behavior.
3. **The agreement eval exposed the stub.** Spearman 0.17 between the ML
   ranking and the keyless scorecard stub says the stub is a weak second
   opinion — which is the point: the harness measures the LLM path instead
   of assuming it helps.

Full tables: [`evals/report.md`](evals/report.md).

## Design decisions

1. **Evals as readable code, not a framework.** Rejected: wiring in an eval
   library. Seven functions returning dicts are auditable in one sitting;
   the abstraction cost buys nothing at this scale.
2. **The groundedness gate is code, not prompt.** `verify_citations()`
   rejects any rationale citing a figure absent from the application —
   same testable-gate design as my econ-rag-agent's `verify_numbers()`.
   Rejected: asking the model nicely.
3. **Protected-adjacent columns are slice-only.** `age`,
   `personal_status_sex`, `foreign_worker` never enter the feature set —
   a test enforces this — but the harness evaluates across them, because
   you audit exactly the attributes you refuse to model on.
4. **Keyless by default.** The LLM path without a key is a deterministic
   scorecard stub, so every eval runs offline and reproducibly. Rejected:
   requiring an API key to run at all. The live Claude path ships
   **UNTESTED-LIVE** until its first keyed run.

## Data provenance

Statlog German Credit: 1,000 **real** loan applications with **real**
repayment outcomes (UCI, CC BY 4.0) — the first repo in this set with
genuine outcome labels rather than synthetic data. It is also ~50 years
old and 1,000 rows: honest substrate for evaluating an evaluation harness,
not a production credit model. Details and caveats:
[`data/PROVENANCE.md`](data/PROVENANCE.md).

---

