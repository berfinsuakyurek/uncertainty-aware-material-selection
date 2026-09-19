"""Utilities for uncertainty-aware material selection."""

import numpy as np
import pandas as pd


def identify_pareto_optimal(
    df,
    performance_col="predicted_K_VRH",
    uncertainty_col="uncertainty"
):
    """
    Identify Pareto-optimal candidates.

    Higher performance and lower uncertainty are preferred.
    """

    predictions = df[performance_col].to_numpy()
    uncertainties = df[uncertainty_col].to_numpy()

    pareto_mask = []

    for i in range(len(df)):
        dominated = (
            (predictions >= predictions[i])
            & (uncertainties <= uncertainties[i])
            & (
                (predictions > predictions[i])
                | (uncertainties < uncertainties[i])
            )
        )

        pareto_mask.append(not dominated.any())

    return np.array(pareto_mask)


def select_decision_profiles(
    pareto_df,
    performance_col="predicted_K_VRH",
    uncertainty_col="uncertainty"
):
    """
    Select high-performance, balanced, and low-risk candidates
    from a Pareto-optimal candidate set.

    The balanced profile assigns equal normalized importance
    to performance and reliability.
    """

    candidates = pareto_df.copy()

    performance_range = (
        candidates[performance_col].max()
        - candidates[performance_col].min()
    )

    uncertainty_range = (
        candidates[uncertainty_col].max()
        - candidates[uncertainty_col].min()
    )

    if performance_range == 0 or uncertainty_range == 0:
        raise ValueError(
            "Performance and uncertainty must have non-zero ranges."
        )

    candidates["performance_norm"] = (
        candidates[performance_col]
        - candidates[performance_col].min()
    ) / performance_range

    candidates["reliability_norm"] = 1 - (
        (
            candidates[uncertainty_col]
            - candidates[uncertainty_col].min()
        )
        / uncertainty_range
    )

    candidates["balanced_score"] = (
        0.5 * candidates["performance_norm"]
        + 0.5 * candidates["reliability_norm"]
    )

    high_performance = candidates.loc[
        candidates[performance_col].idxmax()
    ]

    balanced = candidates.loc[
        candidates["balanced_score"].idxmax()
    ]

    low_risk = candidates.loc[
        candidates[uncertainty_col].idxmin()
    ]

    return pd.DataFrame([
        {
            "Profile": "High Performance",
            **high_performance.to_dict()
        },
        {
            "Profile": "Balanced",
            **balanced.to_dict()
        },
        {
            "Profile": "Low Risk",
            **low_risk.to_dict()
        }
    ])