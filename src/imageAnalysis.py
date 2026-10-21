import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from imageFeatures import resolve_image_path
from textFeatures import load_data

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

def mean_colour_and_brightness(path):
    with Image.open(resolve_image_path(path)) as img:
        rgb = np.asarray(img.convert("RGB"))
        gray = np.asarray(img.convert("L"))
    r = rgb[:, :, 0].mean()
    g = rgb[:, :, 1].mean()
    b = rgb[:, :, 2].mean()
    brightness = gray.mean()
    return r, g, b, brightness

def main():
    df = load_data()

    rows = []
    for path in df["img_path"]:
        rows.append(mean_colour_and_brightness(path))

    stats = pd.DataFrame(rows, columns=["red", "green", "blue", "brightness"])
    stats["label"] = df["label"].values

    means = stats.groupby("label")[["red", "green", "blue"]].mean()
    print(means)
    print(stats.groupby("label")["brightness"].mean())

    RESULTS_DIR.mkdir(exist_ok=True)
    means.rename(index={0: "Nonfiction", 1: "Fiction"}).plot(
        kind="bar", color=["red", "green", "blue"]
    )
    plt.ylabel("Mean RGB value")
    plt.title("Mean colour by class")
    plt.xticks(rotation=0)
    plt.savefig(RESULTS_DIR / "image_colour_by_class.png")
    plt.close()

    brightness_by_class = [
        stats.loc[stats["label"] == label, "brightness"]
        for label in (0, 1)
    ]
    plt.boxplot(brightness_by_class, tick_labels=["Nonfiction", "Fiction"])
    plt.ylabel("Mean brightness")
    plt.title("Image brightness by class")
    plt.savefig(RESULTS_DIR / "image_brightness_by_class.png")
    plt.close()


if __name__ == "__main__":
    main()