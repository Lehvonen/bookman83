from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from textFeatures import load_data, book_text

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LABEL_NAMES = {1: "Fiction", 0: "Nonfiction"}


def class_balance(df):
    counts = df["label"].map(LABEL_NAMES).value_counts()
    print("Class balance")
    print(counts.to_string())
    print()


def numeric_summary(df):
    df = df.copy()
    df["desc_length"] = df["description"].fillna("").str.len()
    cols = [c for c in ["price", "rating", "stock", "desc_length"] if c in df.columns]
    summary = df.groupby(df["label"].map(LABEL_NAMES))[cols].mean().round(2)
    print("Mean per class")
    print(summary.to_string())
    print()


def top_words(df, n=15):
    """Most common words per class (English stopwords removed)."""
    vec = CountVectorizer(stop_words="english")
    counts = vec.fit_transform(book_text(df))
    vocab = vec.get_feature_names_out()

    result = {}
    for label, name in LABEL_NAMES.items():
        mask = (df["label"] == label).to_numpy()
        totals = counts[mask].sum(axis=0).A1
        top = totals.argsort()[::-1][:n]
        result[name] = [(vocab[i], int(totals[i])) for i in top]
    return result


def plot_top_words(words):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, pairs) in zip(axes, words.items()):
        labels = [w for w, _ in pairs][::-1]
        values = [c for _, c in pairs][::-1]
        ax.barh(labels, values)
        ax.set_title(f"Most common words: {name}")
        ax.set_xlabel("Count")
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "c_top_words.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def colon_analysis(df):
    df = df.copy()
    df["category"] = df["label"].map(LABEL_NAMES)
    df["has_colon"] = df["title"].fillna("").str.contains(":")

    # Count books with and without colons in each class
    counts = pd.crosstab(
        df["category"],
        df["has_colon"]
    ).reindex(
        index=["Fiction", "Nonfiction"],
        columns=[False, True],
        fill_value=0
    )

    counts.columns = ["Without colon", "With colon"]

    print("Colon usage in book titles")
    print(counts.to_string())
    print()

    # Calculate the percentage of titles with a colon
    percentages = (
        df.groupby("category")["has_colon"]
        .mean()
        .mul(100)
        .round(1)
    )

    print("Percentage of titles containing a colon:")
    print(percentages.to_string())
    print()



if __name__ == "__main__":
    df = load_data()
    class_balance(df)
    numeric_summary(df)
    colon_analysis(df)

    words = top_words(df)
    for name, pairs in words.items():
        print(f"Top words, {name}:", ", ".join(f"{w} ({c})" for w, c in pairs))
    print()
    plot_top_words(words)