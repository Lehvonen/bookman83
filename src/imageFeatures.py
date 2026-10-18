import numpy as np
from PIL import Image
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def histogram_one_image(path):
    with Image.open(path) as img:
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
    with Image.open(path) as img:
        gray = img.convert("L").resize((32, 32))
        arr = np.asarray(gray)
    return arr.flatten() / 255


def load_gray_pixels(df):
    rows = []
    for path in df["img_path"]:
        rows.append(gray_vector_one_image(path))
    return np.array(rows)


def grayscale_pca():
    return make_pipeline(
        FunctionTransformer(load_gray_pixels), PCA(n_components=5)
    )
