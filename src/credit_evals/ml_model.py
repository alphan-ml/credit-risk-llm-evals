"""The ML path: gradient boosting -> probability of default.

Boring on purpose. The repo's subject is the eval harness; the model
just needs to be a competent, reproducible baseline (fixed seed, fixed
split) so eval numbers are stable run to run.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, MODEL_FEATURES, NUMERIC, TARGET

SEED = 20260726


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    test_index: np.ndarray  # row ids back into the full frame


def make_split(df: pd.DataFrame, test_size: float = 0.3) -> Split:
    X = df[MODEL_FEATURES]
    y = df[TARGET].to_numpy()
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
        X, y, idx, test_size=test_size, random_state=SEED, stratify=y)
    return Split(X_tr, X_te, y_tr, y_te, idx_te)


def train(split: Split) -> Pipeline:
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ])),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ])
    pipe.fit(split.X_train, split.y_train)
    return pipe


def predict_pd(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
