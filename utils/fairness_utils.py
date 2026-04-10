"""
Fairness utility functions for ML Fairness Studio.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equal_opportunity_difference,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_REDUCTIONS_AVAILABLE = True
except ImportError:
    FAIRLEARN_REDUCTIONS_AVAILABLE = False

try:
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_POST_AVAILABLE = True
except ImportError:
    FAIRLEARN_POST_AVAILABLE = False


def compute_group_metrics(y_true, y_pred, sensitive_features):
    """
    Compute per-group performance metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    sensitive_features : array-like or pd.Series

    Returns
    -------
    pd.DataFrame with columns: Group, Count, Accuracy, Precision, Recall, F1, Selection Rate
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive_features)

    groups = np.unique(sensitive)
    rows = []

    # Determine positive label (last class when sorted)
    unique_labels = np.unique(y_true)
    pos_label = unique_labels[-1] if len(unique_labels) > 0 else 1

    for group in groups:
        mask = sensitive == group
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        count = mask.sum()

        acc = accuracy_score(yt, yp)
        try:
            prec = precision_score(yt, yp, pos_label=pos_label, zero_division=0)
            rec = recall_score(yt, yp, pos_label=pos_label, zero_division=0)
            f1 = f1_score(yt, yp, pos_label=pos_label, zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0
        selection_rate = (yp == pos_label).mean()

        rows.append({
            "Group": str(group),
            "Count": int(count),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Selection Rate": round(selection_rate, 4),
        })

    return pd.DataFrame(rows)


def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    """
    Compute standard fairness metrics.

    Returns a dict with keys:
      - disparate_impact
      - statistical_parity_difference
      - demographic_parity_difference
      - demographic_parity_ratio
      - equalized_odds_difference
      - equal_opportunity_difference
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive_features)

    unique_labels = np.unique(y_true)
    pos_label = unique_labels[-1] if len(unique_labels) > 0 else 1

    metrics = {}

    # ── Manual group-based metrics ──────────────────────────────────────────
    groups = np.unique(sensitive)
    selection_rates = {}
    for g in groups:
        mask = sensitive == g
        if mask.sum() == 0:
            selection_rates[g] = 0.0
        else:
            selection_rates[g] = float((y_pred[mask] == pos_label).mean())

    rates = list(selection_rates.values())
    if len(rates) >= 2:
        max_rate = max(rates)
        min_rate = min(rates)
        metrics["statistical_parity_difference"] = round(max_rate - min_rate, 4)
        metrics["disparate_impact"] = round(min_rate / max_rate, 4) if max_rate > 0 else 0.0
    else:
        metrics["statistical_parity_difference"] = 0.0
        metrics["disparate_impact"] = 1.0

    # ── Fairlearn metrics (more precise) ────────────────────────────────────
    if FAIRLEARN_AVAILABLE:
        try:
            metrics["demographic_parity_difference"] = round(
                demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive), 4
            )
        except Exception as e:
            metrics["demographic_parity_difference"] = metrics.get("statistical_parity_difference", 0.0)

        try:
            metrics["demographic_parity_ratio"] = round(
                demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive), 4
            )
        except Exception:
            max_r = max(rates) if rates else 1
            min_r = min(rates) if rates else 1
            metrics["demographic_parity_ratio"] = round(min_r / max_r, 4) if max_r > 0 else 1.0

        try:
            metrics["equalized_odds_difference"] = round(
                equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive), 4
            )
        except Exception:
            metrics["equalized_odds_difference"] = None

        try:
            metrics["equal_opportunity_difference"] = round(
                equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive), 4
            )
        except Exception:
            metrics["equal_opportunity_difference"] = None
    else:
        # Fallback manual calculations
        metrics["demographic_parity_difference"] = metrics.get("statistical_parity_difference", 0.0)
        max_r = max(rates) if rates else 1
        min_r = min(rates) if rates else 1
        metrics["demographic_parity_ratio"] = round(min_r / max_r, 4) if max_r > 0 else 1.0

        # Equalized odds: max difference in TPR and FPR across groups
        tpr_dict = {}
        fpr_dict = {}
        for g in groups:
            mask = sensitive == g
            if mask.sum() == 0:
                continue
            yt_g = y_true[mask]
            yp_g = y_pred[mask]
            pos_mask = yt_g == pos_label
            neg_mask = yt_g != pos_label
            tpr_dict[g] = float((yp_g[pos_mask] == pos_label).mean()) if pos_mask.sum() > 0 else 0.0
            fpr_dict[g] = float((yp_g[neg_mask] == pos_label).mean()) if neg_mask.sum() > 0 else 0.0

        if len(tpr_dict) >= 2:
            tpr_diff = max(tpr_dict.values()) - min(tpr_dict.values())
            fpr_diff = max(fpr_dict.values()) - min(fpr_dict.values())
            metrics["equalized_odds_difference"] = round(max(tpr_diff, fpr_diff), 4)
            metrics["equal_opportunity_difference"] = round(tpr_diff, 4)
        else:
            metrics["equalized_odds_difference"] = 0.0
            metrics["equal_opportunity_difference"] = 0.0

    return metrics


def apply_reweighing(X, y, sensitive_features):
    """
    Apply reweighing pre-processing bias mitigation.
    Returns sample_weights array that upweights under-represented group/label combos.
    """
    # Convert to string dtype to avoid sort errors when mixing strings with NaN floats
    y = pd.array(y).astype(str)
    y = np.where(y == "nan", "Unknown", y)

    sensitive = pd.array(sensitive_features).astype(str)
    sensitive = np.where(sensitive == "nan", "Unknown", sensitive)

    n = len(y)
    weights = np.ones(n)

    unique_labels = np.unique(y)
    unique_groups = np.unique(sensitive)

    # Expected probability: P(Y=y) * P(S=s)
    # Observed probability: P(Y=y, S=s)
    label_probs = {lbl: (y == lbl).mean() for lbl in unique_labels}
    group_probs = {grp: (sensitive == grp).mean() for grp in unique_groups}

    for lbl in unique_labels:
        for grp in unique_groups:
            mask = (y == lbl) & (sensitive == grp)
            observed = mask.mean()
            expected = label_probs[lbl] * group_probs[grp]
            if observed > 0:
                w = expected / observed
                weights[mask] = w

    # Normalize so that weights sum to n
    weights = weights / weights.mean()
    return weights


def apply_fairness_constraint(model, X_train, y_train, sensitive_train,
                               constraint="demographic_parity", eps=0.05):
    """
    Apply ExponentiatedGradient in-processing fairness constraint.

    Parameters
    ----------
    model : sklearn estimator
    constraint : 'demographic_parity' or 'equalized_odds'
    eps : fairness constraint tolerance

    Returns
    -------
    Fitted ExponentiatedGradient wrapper or original model on failure
    """
    if not FAIRLEARN_REDUCTIONS_AVAILABLE:
        raise ImportError("fairlearn is required for ExponentiatedGradient. Install it with: pip install fairlearn")

    if constraint == "demographic_parity":
        constraint_obj = DemographicParity(difference_bound=eps)
    elif constraint == "equalized_odds":
        constraint_obj = EqualizedOdds(difference_bound=eps)
    else:
        constraint_obj = DemographicParity(difference_bound=eps)

    mitigator = ExponentiatedGradient(model, constraint_obj)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator


def apply_threshold_optimizer(model, X_train, y_train, sensitive_train,
                               X_test, sensitive_test, constraint="demographic_parity"):
    """
    Apply ThresholdOptimizer post-processing fairness mitigation.

    Returns fitted ThresholdOptimizer.
    """
    if not FAIRLEARN_POST_AVAILABLE:
        raise ImportError("fairlearn is required for ThresholdOptimizer. Install it with: pip install fairlearn")

    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        objective="accuracy_score",
        predict_method="predict_proba",
    )
    optimizer.fit(X_train, y_train, sensitive_features=sensitive_train)
    return optimizer


def get_fairness_interpretation(metric_name, value):
    """
    Return a human-readable interpretation of a fairness metric value.
    """
    if value is None:
        return "N/A", "gray"

    interpretations = {
        "demographic_parity_difference": {
            "good": (abs(value) < 0.1, f"|{value:.3f}| < 0.1 is considered fair"),
            "bad": (abs(value) >= 0.1, f"|{value:.3f}| ≥ 0.1 indicates potential bias"),
        },
        "demographic_parity_ratio": {
            "good": (value >= 0.8, f"{value:.3f} ≥ 0.8 passes the 80% rule"),
            "bad": (value < 0.8, f"{value:.3f} < 0.8 fails the 80% rule (disparate impact)"),
        },
        "equalized_odds_difference": {
            "good": (abs(value) < 0.1, f"|{value:.3f}| < 0.1 is considered fair"),
            "bad": (abs(value) >= 0.1, f"|{value:.3f}| ≥ 0.1 indicates potential bias"),
        },
        "equal_opportunity_difference": {
            "good": (abs(value) < 0.1, f"|{value:.3f}| < 0.1 is considered fair"),
            "bad": (abs(value) >= 0.1, f"|{value:.3f}| ≥ 0.1 indicates potential bias"),
        },
        "disparate_impact": {
            "good": (value >= 0.8, f"{value:.3f} ≥ 0.8 passes the 80% rule"),
            "bad": (value < 0.8, f"{value:.3f} < 0.8 fails the 80% rule"),
        },
    }

    if metric_name in interpretations:
        info = interpretations[metric_name]
        if info["good"][0]:
            return info["good"][1], "green"
        else:
            return info["bad"][1], "red"
    return f"Value: {value:.3f}", "gray"
