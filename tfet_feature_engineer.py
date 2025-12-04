#!/usr/bin/env python3
"""
Reusable TFET feature engineering utilities shared by training and serving.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TFETFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature generator used by the TFET RandomForest model."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["Lov", "tn", "tox", "WK", "VG"])

        df = X.copy()
        df["Lov_pow"] = df["Lov"] ** 0.85
        df["exp_neg_tn"] = np.exp(-0.45 * df["tn"])
        df["exp_neg_tox"] = np.exp(-0.28 * (df["tox"] - 1.0))
        df["VG_sq"] = df["VG"] ** 2
        df["Lov_VG"] = df["Lov"] * df["VG"]
        df["Lov_VG_pow"] = (df["Lov"] ** 0.85) * df["VG"]
        df["WK_shift"] = df["WK"] - 4.2

        cols = [
            "Lov",
            "tn",
            "tox",
            "WK",
            "VG",
            "Lov_pow",
            "exp_neg_tn",
            "exp_neg_tox",
            "VG_sq",
            "Lov_VG",
            "Lov_VG_pow",
            "WK_shift",
        ]

        return df[cols].values


def register_pickle_aliases(*aliases: str) -> None:
    """
    Some previously-saved artifacts reference this class via the historical
    `main.TFETFeatureEngineer` path. Expose aliases so joblib can resolve them.
    """

    module = sys.modules[__name__]
    for alias in aliases:
        sys.modules[alias] = module

