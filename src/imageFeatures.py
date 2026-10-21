import numpy as np
from PIL import Image
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve_image_path(path):
    # the scraper saves img_path="" when a cover download fails
    if not path:
        raise ValueError("Empty img_path: cover image is missing")
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path
    return image_path


def histogram_one_image(path):
    with Image.open(resolve_image_path(path)) as img:
        pixels = np.asarray(img.convert("RGB"))

    parts = []
    for channel in range(3):
        counts, _ = np.histogram(
            pixels[:, :, channel], bins=8, range=(0, 256)
        )
        parts.append(counts / counts.sum())

    return np.concatenate(parts)


def compute_histograms(df):
    rows = []
    for path in df["img_path"]:
        rows.append(histogram_one_image(path))
    return np.array(rows)


def color_histogram():
    return FunctionTransformer(compute_histograms)


def gray_vector_one_image(path):
    with Image.open(resolve_image_path(path)) as img:
        gray = img.convert("L").resize((32, 32))
        arr = np.asarray(gray)
    return arr.flatten() / 255


def load_gray_pixels(df):
    rows = []
    for path in df["img_path"]:
        rows.append(gray_vector_one_image(path))
    return np.array(rows)


def grayscale_pca():
    # 28 is the smallest K that keeps 90% of the variance on the 175 covers
    # (K=5 kept 76%); the contract allows K <= 140, the size of a training fold.
    # random_state fixes the randomized SVD solver sklearn picks for this shape
    return make_pipeline(
        FunctionTransformer(load_gray_pixels),
        PCA(n_components=28, random_state=42),
    )
