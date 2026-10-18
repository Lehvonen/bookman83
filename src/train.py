import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score

RANDOM_STATE = 42


def evaluate(model, X, y):
    """Run 5-fold stratified cross-validation and return mean/std per metric."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics = {
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
    }
    scores = cross_validate(model, X, y, cv=folds, scoring=metrics)

    results = {}
    for metric in metrics:
        fold_scores = scores[f"test_{metric}"]
        results[metric] = {"mean": fold_scores.mean(), "std": fold_scores.std()}
    return results


def print_results(name, results):
    """Print a model's name followed by the mean and std of each metric."""
    print(name)
    for metric, stats in results.items():
        print(f"{metric}: {stats['mean']:.3f} +/- {stats['std']:.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.random((175, 10))
    y = np.array([1] * 65 + [0] * 110)

    results = evaluate(DummyClassifier(strategy="most_frequent"), X, y)
    print_results("Majority baseline", results)

    results = evaluate(LogisticRegression(), X, y)
    print_results("Logistic regression (dummy data)", results)
