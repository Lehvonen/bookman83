import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from textFeatures import tfidf_features

RANDOM_STATE = 42


def get_models():
    """Return a dict of fresh, unfitted models keyed by name."""
    return {
        "Logistic regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Naive Bayes": MultinomialNB(fit_prior=False),
        "Random forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    }


# Temporary until data/books_clean.csv exists (sync point 1).
# Then replace with pd.read_csv("data/books_clean.csv").
def load_raw_data(path="books_data_raw.json"):
    """Load the raw scraped JSON into a DataFrame with a label column."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["label"] = (df["category"] == "Fiction").astype(int)
    return df


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

    for name, model in get_models().items():
        results = evaluate(model, X, y)
        print_results(f"{name} (dummy data)", results)

    df = load_raw_data()
    for name, model in get_models().items():
        pipeline = make_pipeline(tfidf_features(), model)
        results = evaluate(pipeline, df, df["label"])
        print_results(f"{name} + TF-IDF", results)
