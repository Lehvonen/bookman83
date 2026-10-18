import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

def mean_colour_and_brightness(path):
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
        gray = np.asarray(img.convert("L"))
    r = rgb[:, :, 0].mean()
    g = rgb[:, :, 1].mean()
    b = rgb[:, :, 2].mean()
    brightness = gray.mean()
    return r, g, b, brightness


def main():
    df = pd.DataFrame({
        "img_path": [
            os.path.join(project_dir, "test_images", f"{i}.jpg")
            for i in range(10)
        ],
        "label": [1] * 5 + [0] * 5,
    })

    rows = []
    for path in df["img_path"]:
        rows.append(mean_colour_and_brightness(path))

    stats = pd.DataFrame(rows, columns=["red", "green", "blue", "brightness"])
    stats["label"] = df["label"].values

    means = stats.groupby("label")[["red", "green", "blue"]].mean()
    print(means)
    print(stats.groupby("label")["brightness"].mean())

    os.makedirs(results_dir, exist_ok=True)
    means.rename(index={0: "Nonfiction", 1: "Fiction"}).plot(
        kind="bar", color=["red", "green", "blue"]
    )
    plt.ylabel("Mean RGB value")
    plt.title("Mean colour by class")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(results_dir, "image_colour_by_class.png"))
    plt.close()

    brightness_by_class = [
        stats.loc[stats["label"] == label, "brightness"]
        for label in (0, 1)
    ]
    plt.boxplot(brightness_by_class, tick_labels=["Nonfiction", "Fiction"])
    plt.ylabel("Mean brightness")
    plt.title("Image brightness by class")
    plt.savefig(os.path.join(results_dir, "image_brightness_by_class.png"))
    plt.close()


if __name__ == "__main__":
    main()