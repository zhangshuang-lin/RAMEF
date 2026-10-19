"""Measurement-consistent support-guided boundary transport (MC-SBT).

This module deliberately operates on frozen M2 tokens.  It never receives an
unlabelled target label: labels are used only by the experiment runner after a
prediction has been fixed, for audit metrics.  The P0 hypothesis is therefore
precise: target geometry and record-level agreement can repair the IR/OR
boundary without changing the representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BoundaryConfig:
    c: float = 1.0
    support_weight: float = 20.0
    pseudo_weight: float = 1.0
    confidence: float = 0.90
    iterations: int = 3
    seed: int = 0


@dataclass
class BoundaryModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    audit: dict

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.decision_function(
            self.scaler.transform(np.asarray(features, dtype=np.float64))))


def _balanced_consistent_indices(probabilities: np.ndarray,
                                 threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select record pseudo-labels only when every view agrees.

    Args:
        probabilities: [measurement, view, class].
        threshold: minimum confidence of the mean record probability.

    Returns:
        selected measurement indices, pseudo labels and confidences.  The
        selection is class balanced by truncating to the smallest non-zero
        predicted class count; if any class is absent, no pseudo labels are
        returned, preventing majority-class collapse.
    """
    if probabilities.ndim != 3 or probabilities.shape[-1] != 3:
        raise ValueError("probabilities must have shape [measurement, view, 3]")
    view_prediction = probabilities.argmax(-1)
    agreement = np.all(view_prediction == view_prediction[:, :1], axis=1)
    mean_probability = probabilities.mean(1)
    pseudo = mean_probability.argmax(1)
    confidence = mean_probability[np.arange(len(mean_probability)), pseudo]
    eligible = agreement & (confidence >= float(threshold))
    groups = [np.flatnonzero(eligible & (pseudo == label)) for label in range(3)]
    if any(len(group) == 0 for group in groups):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), np.empty(0, dtype=np.float64)
    keep = min(len(group) for group in groups)
    selected = []
    for group in groups:
        order = group[np.argsort(-confidence[group], kind="stable")]
        selected.extend(order[:keep].tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return selected, pseudo[selected].astype(np.int64), confidence[selected]


def fit_boundary(source_features: np.ndarray,
                 source_labels: np.ndarray,
                 support_features: np.ndarray,
                 support_labels: np.ndarray,
                 unlabeled_views: np.ndarray,
                 config: BoundaryConfig,
                 unlabeled_audit_labels: Optional[np.ndarray] = None) -> BoundaryModel:
    """Fit a source-regularised target boundary with record-consistent pseudo labels.

    ``unlabeled_audit_labels`` is never used in fitting or selection.  It is an
    optional source-only diagnostic that reports pseudo-label precision after
    the model is already fitted.
    """
    source_features = np.asarray(source_features, dtype=np.float64)
    support_features = np.asarray(support_features, dtype=np.float64)
    unlabeled_views = np.asarray(unlabeled_views, dtype=np.float64)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    support_labels = np.asarray(support_labels, dtype=np.int64)
    if unlabeled_views.ndim != 3:
        raise ValueError("unlabeled_views must be [measurement, view, feature]")
    # Unlabelled target statistics are legitimate in SSDA.  Fitting the scaler
    # on them aligns only marginal location/scale; no class or test information
    # enters this operation.
    unlabeled_flat = unlabeled_views.reshape(-1, unlabeled_views.shape[-1])
    # A strictly supervised/support-only protocol has no target-unlabelled
    # cohort.  In that case fit and apply the scaler only to the authorised
    # source/support observations; sklearn rejects transform on an empty array.
    scaler_rows = [source_features, support_features]
    if len(unlabeled_flat):
        scaler_rows.append(unlabeled_flat)
    scaler = StandardScaler().fit(np.vstack(scaler_rows))
    sx = scaler.transform(source_features)
    tx = scaler.transform(support_features)
    ux = (scaler.transform(unlabeled_flat).reshape(unlabeled_views.shape)
          if len(unlabeled_flat) else unlabeled_views.copy())

    fit_x = np.vstack([sx, tx])
    fit_y = np.concatenate([source_labels, support_labels])
    fit_w = np.concatenate([np.ones(len(sx)),
                            np.full(len(tx), float(config.support_weight))])
    classifier = LogisticRegression(
        C=float(config.c), class_weight=None, max_iter=4000,
        random_state=int(config.seed), solver="lbfgs",
    ).fit(fit_x, fit_y, sample_weight=fit_w)

    history = []
    selected = np.empty(0, dtype=np.int64)
    pseudo = np.empty(0, dtype=np.int64)
    confidence = np.empty(0, dtype=np.float64)
    for iteration in range(max(0, int(config.iterations))):
        if len(ux) == 0:
            break
        probabilities = classifier.predict_proba(ux.reshape(-1, ux.shape[-1])).reshape(
            ux.shape[0], ux.shape[1], 3)
        selected, pseudo, confidence = _balanced_consistent_indices(
            probabilities, config.confidence)
        counts = {str(label): int(np.sum(pseudo == label)) for label in range(3)}
        history.append({"iteration": iteration, "accepted": int(len(selected)),
                        "accepted_per_class": counts})
        if len(selected) == 0:
            break
        # A record contributes its mean representation once.  This avoids
        # overweighting a measurement merely because more windows were cut.
        pseudo_x = ux[selected].mean(1)
        train_x = np.vstack([sx, tx, pseudo_x])
        train_y = np.concatenate([source_labels, support_labels, pseudo])
        train_w = np.concatenate([
            np.ones(len(sx)), np.full(len(tx), float(config.support_weight)),
            float(config.pseudo_weight) * confidence,
        ])
        classifier = LogisticRegression(
            C=float(config.c), class_weight=None, max_iter=4000,
            random_state=int(config.seed) + iteration + 1, solver="lbfgs",
        ).fit(train_x, train_y, sample_weight=train_w)

    audit = {"history": history, "final_accepted": int(len(selected)),
             "final_accepted_per_class": {
                 str(label): int(np.sum(pseudo == label)) for label in range(3)}}
    if unlabeled_audit_labels is not None and len(selected):
        truth = np.asarray(unlabeled_audit_labels, dtype=np.int64)
        audit["pseudo_label_accuracy_audit_only"] = float(np.mean(pseudo == truth[selected]))
    return BoundaryModel(scaler=scaler, classifier=classifier, audit=audit)


def coarse_safe_prediction(boundary_logits: np.ndarray,
                           m2_logits: np.ndarray) -> np.ndarray:
    """Preserve M2's N/Fault decision and adapt only the IR/OR boundary."""
    boundary_logits = np.asarray(boundary_logits)
    m2_logits = np.asarray(m2_logits)
    if boundary_logits.shape != m2_logits.shape or boundary_logits.shape[1] != 3:
        raise ValueError("boundary_logits and m2_logits must have matching [N,3] shape")
    base = m2_logits.argmax(1)
    prediction = np.where(boundary_logits[:, 1] >= boundary_logits[:, 2], 1, 2)
    prediction[base == 0] = 0
    return prediction.astype(np.int64)


def margin_gated_prediction(boundary_logits: np.ndarray, m2_logits: np.ndarray,
                            threshold: float) -> np.ndarray:
    """Apply a query-local IR/OR switch only above a calibrated margin."""
    boundary_logits, m2_logits = np.asarray(boundary_logits), np.asarray(m2_logits)
    base = m2_logits.argmax(1).astype(np.int64)
    candidate = coarse_safe_prediction(boundary_logits, m2_logits)
    margin = np.abs(boundary_logits[:, 1] - boundary_logits[:, 2])
    switch = (base != 0) & (candidate != base) & (margin >= float(threshold))
    output = base.copy(); output[switch] = candidate[switch]
    return output


def calibrate_margin_threshold(boundary_logits: np.ndarray, m2_logits: np.ndarray,
                               labels: np.ndarray) -> tuple[float, dict]:
    """Choose the safest switch threshold from labelled support LOO predictions.

    Accuracy is maximised first.  Exact ties prefer fewer boundary changes,
    which implements identity fallback.  No query or unlabelled label is used.
    """
    boundary_logits, m2_logits = np.asarray(boundary_logits), np.asarray(m2_logits)
    labels = np.asarray(labels, dtype=np.int64)
    margin = np.abs(boundary_logits[:, 1] - boundary_logits[:, 2])
    candidate = coarse_safe_prediction(boundary_logits, m2_logits)
    base = m2_logits.argmax(1)
    changed_margin = np.unique(margin[(base != 0) & (candidate != base)])
    thresholds = [0.0] + [float(x + 1e-9) for x in changed_margin] + [float("inf")]
    rows = []
    for threshold in thresholds:
        prediction = margin_gated_prediction(boundary_logits, m2_logits, threshold)
        rows.append({"threshold": threshold,
                     "accuracy": float(np.mean(prediction == labels)),
                     "changes": int(np.sum(prediction != base)),
                     "new_errors": int(np.sum((base == labels) & (prediction != labels))),
                     "fixes": int(np.sum((base != labels) & (prediction == labels)))})
    chosen = max(rows, key=lambda x: (x["accuracy"], -x["new_errors"],
                                      -x["changes"], x["threshold"]))
    return float(chosen["threshold"]), {"chosen": chosen, "candidates": rows}
