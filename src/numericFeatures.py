import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def numeric_columns(df):
    desc_len = df["description"].str.len()

    return np.column_stack([
        df["price"],
        df["rating"],
        df["stock"],
        desc_len
    ]).astype(float)

def numeric_features():
    return make_pipeline(
        FunctionTransformer(numeric_columns, validate=False),
        StandardScaler()
    )

if __name__ == "__main__":
    from textFeatures import load_data

    df = load_data()

    X = numeric_features().fit_transform(df)

    print(X.shape)  # should be (175, 4)
    print(X[:5])    # show the first five books