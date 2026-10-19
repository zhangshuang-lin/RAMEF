"""Target-speed evidence re-anchoring with diagonal Student-t inference."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


KAPPAS = (0.0, 0.25, 1.0, 4.0)


def posterior_parameters(source_x: np.ndarray, source_y: np.ndarray,
                         support_x: np.ndarray, support_y: np.ndarray,
                         kappa: float, classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Class-specific target posterior with a source distribution prior."""
    means, variances = [], []
    for cls in range(classes):
        source = source_x[source_y == cls]
        target = support_x[support_y == cls]
        if len(source) == 0 or len(target) == 0:
            raise ValueError("Every class requires source and target support")
        source_mean = source.mean(0)
        source_var = source.var(0, ddof=1) + 0.05
        target_mean = target.mean(0)
        target_var = target.var(0, ddof=1) + 0.05 if len(target) > 1 else source_var
        denominator = kappa + len(target)
        mean = (kappa * source_mean + len(target) * target_mean) / denominator
        # Pooled within-distribution variance plus the explicit source-target
        # displacement term.  This prevents confident extrapolation when the
        # evidence mechanism changes with speed.
        between = (kappa * len(target) / max(denominator, 1e-12)) * (source_mean - target_mean) ** 2
        numerator = kappa * source_var + max(len(target) - 1, 1) * target_var + between
        variance = numerator / max(kappa + max(len(target) - 1, 1), 1e-12)
        means.append(mean); variances.append(np.maximum(variance, 0.05))
    return np.stack(means), np.stack(variances)


def student_t_logits(query: np.ndarray, means: np.ndarray, variances: np.ndarray,
                     degrees: float = 5.0) -> np.ndarray:
    """Independent-dimension Student-t log likelihood, up to class constants."""
    diff2 = (query[:, None, :] - means[None, :, :]) ** 2
    scale = variances[None, :, :]
    constant = gammaln((degrees + 1.0) / 2.0) - gammaln(degrees / 2.0)
    value = (constant - 0.5 * np.log(degrees * np.pi * scale)
             - 0.5 * (degrees + 1.0) * np.log1p(diff2 / (degrees * scale)))
    return value.sum(axis=2)


def predict(source_x: np.ndarray, source_y: np.ndarray, support_x: np.ndarray,
            support_y: np.ndarray, query_x: np.ndarray, kappa: float) -> np.ndarray:
    means, variances = posterior_parameters(source_x, source_y, support_x, support_y, kappa)
    return student_t_logits(query_x, means, variances).argmax(1)


def choose_kappa_nested(source_x: np.ndarray, source_y: np.ndarray,
                        support_x: np.ndarray, support_y: np.ndarray) -> tuple[float, list[dict]]:
    """Support-only inner LOO; healthy safety precedes headline accuracy."""
    rows = []
    for kappa in KAPPAS:
        guesses = []
        for held in range(len(support_y)):
            keep = np.asarray([i for i in range(len(support_y)) if i != held])
            guesses.append(int(predict(source_x, source_y, support_x[keep], support_y[keep],
                                       support_x[held:held + 1], kappa)[0]))
        guesses = np.asarray(guesses)
        healthy_to_fault = int(np.sum((support_y == 0) & (guesses != 0)))
        accuracy = float(np.mean(guesses == support_y))
        recalls = [float(np.mean(guesses[support_y == c] == c)) for c in range(3)]
        f1s = []
        for c in range(3):
            tp = np.sum((guesses == c) & (support_y == c))
            fp = np.sum((guesses == c) & (support_y != c))
            fn = np.sum((guesses != c) & (support_y == c))
            f1s.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
        rows.append({"kappa": kappa, "healthy_to_fault": healthy_to_fault,
                     "accuracy": accuracy, "macro_f1": float(np.mean(f1s)),
                     "recall_OR": recalls[2]})
    best = max(rows, key=lambda row: (
        -row["healthy_to_fault"], row["accuracy"], row["macro_f1"],
        row["recall_OR"], row["kappa"]))
    return float(best["kappa"]), rows
