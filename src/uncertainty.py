"""Utilities for estimating Random Forest predictive uncertainty."""

import numpy as np


def predict_with_uncertainty(model, X):
    """
    Generate Random Forest predictions and tree-disagreement uncertainty.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted pipeline containing a preprocessing step named
        'preprocessor' and a RandomForestRegressor named 'regressor'.

    X : pandas.DataFrame
        Input material descriptors.

    Returns
    -------
    mean_prediction : numpy.ndarray
        Mean prediction across all trees.

    uncertainty : numpy.ndarray
        Standard deviation of individual tree predictions.

    tree_predictions : numpy.ndarray
        Predictions from every individual tree.
    """

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    X_processed = preprocessor.transform(X)

    tree_predictions = np.array([
        tree.predict(X_processed)
        for tree in regressor.estimators_
    ])

    mean_prediction = tree_predictions.mean(axis=0)
    uncertainty = tree_predictions.std(axis=0)

    return mean_prediction, uncertainty, tree_predictions