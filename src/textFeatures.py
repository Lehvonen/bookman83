from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_data(path=PROJECT_DIR / "data" / "books_clean.csv"):
    return pd.read_csv(path)


def book_text(df):
    return (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()


def tfidf_features():
    return make_pipeline(
        FunctionTransformer(book_text),
        TfidfVectorizer(stop_words="english"),
    )


if __name__ == "__main__":
    df = load_data()
    X = tfidf_features().fit_transform(df)
    print(X.shape)   # should be (175, N)
