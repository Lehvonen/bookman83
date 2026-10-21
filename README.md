# bookman83: Fiction vs Nonfiction Classifier

Classifies books from [books.toscrape.com](https://books.toscrape.com) as Fiction or Nonfiction
from their description, cover image and metadata (price, rating, stock).
175 books: 65 Fiction, 110 Nonfiction.

## Install

Requires Python 3.10+.

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run

Run every command from the project root.

1. `python src/scraper.py`: scrapes the site into `books_data_raw.json` and downloads the covers to `data/images/`.
   *Optional:* the scraped data is already committed, so you can skip this step.
2. `python src/cleanData.py`: writes `data/books_clean.csv`.
3. `python src/train.py`: cross-validates every feature set × model and writes
   `results/results_table.csv`, `results/confusion_matrix.png`, `results/misclassified.csv` and `results/top_words.csv`.

Exploration (optional):

- `python src/textAnalysis.py`: class balance, numeric summaries, top words (`results/c_top_words.png`)
- `python src/imageAnalysis.py`: cover colour and brightness by class

## Project layout

| Path | Contents |
|---|---|
| `src/scraper.py`, `src/cleanData.py` | Data collection and cleaning |
| `src/textFeatures.py`, `src/numericFeatures.py`, `src/imageFeatures.py` | Feature functions (unfitted scikit-learn transformers) |
| `src/train.py` | Stratified 5-fold CV, results table, best-model inspection |
| `src/textAnalysis.py`, `src/imageAnalysis.py` | Exploratory analysis |
| `data/` | Cleaned CSV and cover images |
| `results/` | Tables and figures |
