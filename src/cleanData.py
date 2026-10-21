import re
import json
import pandas as pd
from PIL import Image
from pathlib import Path


def clean_description(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*\.\.\.more$", "", text)
    probe = text[:40]
    idx = text.find(probe, 1)
    if idx > 0:
        text = text[idx:]
    return text


PROJECT_DIR = Path(__file__).resolve().parents[1]

with (PROJECT_DIR / "books_data_raw.json").open(encoding="utf-8") as f:
    records = json.load(f)
df = pd.DataFrame(records)

df["description"] = df["description"].apply(clean_description)
df["price"] = df["price"].str.replace(r"[^\d.]", "", regex=True).astype(float)
df["rating"] = df["rating"].astype(int)
df["stock"] = df["stock"].astype(int)
df["label"] = (df["category"] == "Fiction").astype(int)

# match the shared contract: rename img and set column order
df = df.rename(columns={"img": "img_url"})
df = df[["upc", "title", "label", "description", "price", "rating", "stock", "img_url", "img_path"]]

# checks
print(df.isna().sum())
print("empty descriptions:", (df["description"] == "").sum())
print("duplicate UPCs:", df["upc"].duplicated().sum())
print("duplicate titles:", df["title"].duplicated().sum())

bad_images = []
for _, row in df.iterrows():
    try:
        with Image.open(PROJECT_DIR / row["img_path"]) as im:
            im.verify()
    except Exception as e:
        bad_images.append((row["upc"], str(e)))
print("unreadable images:", bad_images)

# save
df.to_csv(PROJECT_DIR / "data" / "books_clean.csv", index=False, encoding="utf-8")
print(df.shape)
print(df["label"].value_counts())