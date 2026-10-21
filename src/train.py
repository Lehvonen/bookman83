from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from textFeatures import tfidf_features, load_data
from numericFeatures import numeric_features
from imageFeatures import color_histogram, grayscale_pca

RANDOM_STATE = 42

# MultinomialNB fails on negative values, so it only runs on these feature sets.
# StandardScaler (numeric features) and PCA (grayscale) produce negative values.
NON_NEGATIVE_FEATURES = {"TF-IDF", "Colour histogram"}

# Must match keys in get_feature_sets() and get_models().
# TF-IDF + logistic regression is inspected because its coefficients give readable
# word weights. Numeric + logistic regression scores higher (see results_table.csv).
BEST_FEATURES = "TF-IDF"
BEST_MODEL = "Logistic regression"


def get_models():
    """Return a dict of fresh, unfitted models keyed by name."""
    return {
        "Logistic regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Naive Bayes": MultinomialNB(fit_prior=False),
        "Random forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    }


def get_feature_sets():
    """Return a dict of feature functions (not called) keyed by name."""
    return {
        "TF-IDF": tfidf_features,
        "Numeric": numeric_features,
        "Colour histogram": color_histogram,
        "Grayscale PCA": grayscale_pca,
    }


def get_folds():
    """Return the stratified 5-fold split shared by every evaluation."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(model, X, y):
    """Run 5-fold stratified cross-validation and return mean/std per metric."""
    metrics = {
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
    }
    scores = cross_validate(model, X, y, cv=get_folds(), scoring=metrics)

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


def make_row(features, model_name, results):
    """Flatten one evaluation into a table row with a mean and std column per metric."""
    row = {"features": features, "model": model_name}
    for metric, stats in results.items():
        row[metric] = stats["mean"]
        row[f"{metric}_std"] = stats["std"]
    return row


def results_table(df):
    """Evaluate the baseline and every feature set x model combination."""
    rows = []

    results = evaluate(DummyClassifier(strategy="most_frequent"), df, df["label"])
    rows.append(make_row("-", "Majority baseline", results))

    for feature_name, feature_fn in get_feature_sets().items():
        for model_name, model in get_models().items():
            if model_name == "Naive Bayes" and feature_name not in NON_NEGATIVE_FEATURES:
                continue
            pipeline = make_pipeline(feature_fn(), model)
            results = evaluate(pipeline, df, df["label"])
            rows.append(make_row(feature_name, model_name, results))

    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def inspect_best_model(df):
    """Confusion matrix, misclassified books and top words for the best combination."""
    pipeline = make_pipeline(get_feature_sets()[BEST_FEATURES](), get_models()[BEST_MODEL])

    # Out-of-fold predictions: every book is predicted by a model that never saw it.
    preds = cross_val_predict(pipeline, df, df["label"], cv=get_folds())

    ConfusionMatrixDisplay.from_predictions(
        df["label"], preds, display_labels=["Nonfiction", "Fiction"]
    )
    plt.title(f"{BEST_FEATURES} + {BEST_MODEL}")
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    is_wrong = preds != df["label"]
    wrong = df.loc[is_wrong, ["title"]].copy()
    wrong["actual"] = np.where(df.loc[is_wrong, "label"] == 1, "Fiction", "Nonfiction")
    wrong["predicted"] = np.where(preds[is_wrong] == 1, "Fiction", "Nonfiction")
    print(f"\n{len(wrong)} misclassified books")
    wrong.to_csv("results/misclassified.csv", index=False)

    # Top words come from a model fitted on all the data, since they describe it rather than score it.
    if BEST_FEATURES == "TF-IDF" and BEST_MODEL == "Logistic regression":
        pipeline.fit(df, df["label"])
        vectorizer = pipeline[0][-1]
        words = vectorizer.get_feature_names_out()
        coefs = pipeline[-1].coef_[0]
        order = np.argsort(coefs)
        fiction = order[-15:][::-1]
        nonfiction = order[:15]

        top_words = pd.DataFrame({
            "fiction_word": words[fiction],
            "fiction_weight": coefs[fiction],
            "nonfiction_word": words[nonfiction],
            "nonfiction_weight": coefs[nonfiction],
        })
        print("\nTop words")
        print(top_words.round(3).to_string(index=False))
        top_words.to_csv("results/top_words.csv", index=False)
    else:
        print("Top words are only available for TF-IDF + Logistic regression.")


if __name__ == "__main__":
    df = load_data()
    table = results_table(df)
    print(table.round(3).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/results_table.csv", index=False)

    inspect_best_model(df)
