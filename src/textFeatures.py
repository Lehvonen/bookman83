import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_data(path=None):
    data_path = PROJECT_DIR / "books_data_raw.json" if path is None else Path(path)
    if not data_path.is_absolute():
        data_path = PROJECT_DIR / data_path
    with data_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for book in raw:
        category = book["category"]
        title = book["title"]

        records.append({
            "title": title,
            "category": category,
            "label": 1 if category.lower() == "fiction" else 0,
            "description": book.get("description", ""),
            "price": float(str(book["price"]).replace("£", "")),
            "rating": int(book["rating"]),
            "stock": int(book["stock"]),
            "img_url": book.get("img_url", book.get("img")),
        })

    return pd.DataFrame(records)


def book_text(df):
    return (df["title"] + " " + df["description"]).tolist()


def tfidf_features():
    return make_pipeline(
        FunctionTransformer(book_text),
        TfidfVectorizer(stop_words="english"),
    )


if __name__ == "__main__":
    df = load_data()
    X = tfidf_features().fit_transform(df)
    print(X.shape)   # should be (175, N)