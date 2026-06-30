"""Configuration file for paths and project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# MovieLens 32M dataset (ml-32m/), sitting next to the project folder.
# 32,000,204 ratings · 200,948 users · 84,432 rated movies · half-star scale 0.5-5.0
DATASET_DIR = PROJECT_ROOT.parent / "ml-32m"
RATINGS_PATH = DATASET_DIR / "ratings.csv"
TAGS_PATH = DATASET_DIR / "tags.csv"  # optional content enrichment

# movies.csv: use the full dataset's copy when present (local), otherwise fall
# back to the small bundled copy in data/ — this lets the deployed app (and the
# cached processed set in data/processed/) run WITHOUT the 877 MB ratings file.
ITEMS_PATH = (DATASET_DIR / "movies.csv") if (DATASET_DIR / "movies.csv").exists() \
    else (DATA_DIR / "movies.csv")

# Default recommendation settings.
USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"
TITLE_COL = "title"
GENRES_COL = "genres"

TOP_K = 10
RANDOM_STATE = 42

# --- Preprocessing / sampling settings ---------------------------------------
# MovieLens 32M is very large (32M ratings). EDA is computed on the full dataset,
# but for fast model training/evaluation we apply standard k-core filtering, an
# optional user subsample and an item cap. All of this is documented in the report.
MIN_USER_RATINGS = 20      # keep users with at least this many ratings
MIN_ITEM_RATINGS = 100     # keep items with at least this many ratings (32M is dense)
MAX_USERS = 10000          # subsample of users for modelling (None = keep all)
MAX_ITEMS = 3000           # cap catalog to the most popular items for CF/MF (None = all)
MIN_ITEM_RATINGS_AFTER_SAMPLE = 10  # re-prune thin items after user subsample
RELEVANCE_THRESHOLD = 4.0  # a held-out rating >= this value counts as "relevant" (scale 0.5-5)
