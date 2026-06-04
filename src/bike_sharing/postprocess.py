"""Post-prediction corrections for Bike Sharing Demand forecasts.

Duan's smearing estimator corrects the systematic downward bias a
log1p-trained model incurs when its log-space predictions are mapped back to
the count scale: by Jensen's inequality, ``expm1`` of an (approximately)
unbiased log prediction underestimates the conditional mean of the
right-skewed target, which shows up as under-prediction of peak-demand hours.

The correction is multiplicative in original-count space and is computed from
the model's own TRAINING-set log residuals; it is applied on the log-space
prediction, not on the already-inverted output.
"""

from typing import Any

import numpy as np


def compute_smearing_factor(log_residuals: Any) -> float:
    """Duan's smearing factor ``theta = mean(exp(e_i))``.

    ``log_residuals`` are residuals in the log1p training space on the
    *training* rows only: ``log1p(y_train) - log_pred_train``, where
    ``log_pred_train`` is the model's log-space prediction (e.g.
    ``TransformedTargetRegressor.regressor_.predict``), not the inverted
    count-scale output. ``theta`` is not guaranteed ``>= 1`` - it depends on
    the residual distribution.
    """
    e = np.asarray(log_residuals, dtype=float)
    return float(np.mean(np.exp(e)))


def apply_smearing(log_pred: Any, theta: float) -> np.ndarray:
    """Smeared, non-negative count predictions: ``max(theta * exp(log_pred) - 1, 0)``.

    ``log_pred`` is the model's log-space prediction. ``theta`` multiplies
    ``exp(log_pred)`` *before* the ``- 1`` that undoes ``log1p``, so this is
    NOT ``theta * expm1(log_pred)``. Negatives are clipped to 0 (demand cannot
    be negative), matching the project's ``from_log1p`` contract. At
    ``theta == 1`` this reduces to ``expm1(log_pred)``, i.e. a no-op.
    """
    log_pred = np.asarray(log_pred, dtype=float)
    return np.clip(theta * np.exp(log_pred) - 1.0, a_min=0.0, a_max=None)
