import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42


def evaluate(model, X, y):
    """Run 5-fold stratified cross-validation and return mean/std per metric."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics = ["balanced_accuracy", "f1", "precision", "recall"]
    scores = cross_validate(model, X, y, cv=folds, scoring=metrics)

    results = {}
    for metric in metrics:
        fold_scores = scores[f"test_{metric}"]
        results[metric] = {"mean": fold_scores.mean(), "std": fold_scores.std()}
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.random((175, 10))
    y = np.array([1] * 65 + [0] * 110)

    results = evaluate(LogisticRegression(), X, y)
    for metric, stats in results.items():
        print(f"{metric}: {stats['mean']:.3f} +/- {stats['std']:.3f}")
