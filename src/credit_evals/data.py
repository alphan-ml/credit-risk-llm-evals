"""Load the Statlog German Credit data and split features by role.

Roles matter more than columns here: `MODEL_FEATURES` go into the
classifier, `SLICE_ONLY` columns (age, personal_status_sex,
foreign_worker) are deliberately excluded from modeling and exist only
so the eval harness can ask "how does the system behave across these
groups?" — you cannot audit what you fed into the model as signal.
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "german_credit.csv"

SLICE_ONLY = ["age", "personal_status_sex", "foreign_worker"]
TARGET = "default"

# Everything else is a legitimate application/bureau attribute.
MODEL_FEATURES = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment_duration", "installment_rate",
    "other_debtors", "present_residence", "property",
    "other_installment_plans", "housing", "number_credits", "job",
    "people_liable", "telephone",
]

CATEGORICAL = [
    "status", "credit_history", "purpose", "savings",
    "employment_duration", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone",
]
NUMERIC = [c for c in MODEL_FEATURES if c not in CATEGORICAL]


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Source label: credit_risk 1 = repaid. We predict default.
    df[TARGET] = 1 - df["credit_risk"]
    return df


def render_application(row: pd.Series) -> str:
    """One application as plain text — the LLM ranker's entire view.

    The LLM sees exactly what the model's feature set allows, plus
    nothing from SLICE_ONLY. Keeping this the single serialization
    point is what makes the groundedness gate checkable.
    """
    lines = [f"- {col}: {row[col]}" for col in MODEL_FEATURES]
    return "Loan application:\n" + "\n".join(lines)
