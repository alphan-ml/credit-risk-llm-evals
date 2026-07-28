# Data provenance

`german_credit.csv` is the **Statlog (German Credit Data)** dataset:
1,000 real loan applications from a German bank (1970s), each with the
real repayment outcome. This is REAL historical data, not synthetic —
the first dataset in this portfolio with genuine outcome labels.

- Origin: Prof. Hans Hofmann, UCI Machine Learning Repository
  (https://archive.ics.uci.edu/dataset/144), CC BY 4.0.
- This copy: the selva86/datasets GitHub mirror of the same file with
  human-readable category labels (the raw UCI file uses A11/A12/... codes).
- Label: `credit_risk` — 1 = repaid (good), 0 = defaulted (bad).
  We model `default = 1 - credit_risk`. 300 of 1,000 are defaults.
- The dataset ships with an official cost matrix: misclassifying a
  defaulter as good costs 5, the reverse costs 1. The eval harness uses
  exactly these costs rather than inventing its own.

The data is ~50 years old and 1,000 rows; nothing here
claims production-grade credit modeling. It is real-outcome substrate
for evaluating a scoring system, which is this repo's actual subject.
The `personal_status_sex`, `age`, and `foreign_worker` columns exist in
the source; they are EXCLUDED from model features and used only in the
fairness slice evals — see the README's design decisions.
