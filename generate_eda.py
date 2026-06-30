"""Generate EDA figures and a metrics comparison chart into results/figures/.

Run AFTER main.py (which writes results/metrics.csv):
    python generate_eda.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_ratings, load_items

FIG_DIR = config.RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})

ratings = load_ratings()
items = load_items()

# 1. Rating distribution -------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6, 4))
dist = ratings[config.RATING_COL].value_counts().sort_index()
ax.bar(dist.index, dist.values, width=0.7, color="#33617E")
ax.set_xlabel("Rating"); ax.set_ylabel("Count")
ax.set_title("Rating distribution (MovieLens 32M)")
fig.savefig(FIG_DIR / "rating_distribution.png"); plt.close(fig)

# 2. Ratings per user / per item (long tail) ------------------------------ #
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
upc = ratings[config.USER_COL].value_counts().values
ipc = ratings[config.ITEM_COL].value_counts().values
# Clip the per-user histogram at the 99th percentile so the bulk is visible
# (a handful of power-users rate tens of thousands of movies).
clip = np.percentile(upc, 99)
axes[0].hist(np.clip(upc, None, clip), bins=60, color="#3F7A66")
axes[0].set_title("Ratings per user (clipped at p99)")
axes[0].set_xlabel("# ratings"); axes[0].set_ylabel("# users")
axes[1].plot(np.arange(1, len(ipc) + 1), np.sort(ipc)[::-1], color="#8A4B52")
axes[1].set_title("Item popularity (long tail)")
axes[1].set_xlabel("Item rank"); axes[1].set_ylabel("# ratings"); axes[1].set_yscale("log")
fig.savefig(FIG_DIR / "long_tail.png"); plt.close(fig)

# 3. Genre frequency ------------------------------------------------------ #
genres = (items[config.GENRES_COL].fillna("").str.split("|").explode())
genres = genres[genres.str.len() > 0].value_counts().head(15)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(genres.index[::-1], genres.values[::-1], color="#33617E")
ax.set_title("Top genres by # of movies"); ax.set_xlabel("# movies")
fig.savefig(FIG_DIR / "genre_frequency.png"); plt.close(fig)

# 4. Metrics comparison (accuracy vs beyond-accuracy) --------------------- #
metrics = pd.read_csv(config.RESULTS_DIR / "metrics.csv", index_col=0)
order = metrics.sort_values("NDCG@10", ascending=False).index

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
m1 = metrics.loc[order, ["Precision@10", "Recall@10", "NDCG@10"]]
m1.plot(kind="bar", ax=axes[0], color=["#33617E", "#3F7A66", "#8A4B52"], legend=False)
axes[0].set_title("Accuracy metrics @10"); axes[0].set_ylabel("score")
axes[0].set_xticklabels(order, rotation=35, ha="right")
axes[0].set_ylim(0, m1.values.max() * 1.30)  # headroom for the legend
axes[0].legend(loc="upper right", framealpha=0.95, fontsize=9)

m2cov = metrics.loc[order, ["Coverage"]]
ax_nov = axes[1].twinx()
m2cov.plot(kind="bar", ax=axes[1], color=["#33617E"], width=0.4, position=1, legend=False)
metrics.loc[order, "Novelty"].plot(kind="bar", ax=ax_nov, color="#9A7B3F",
                                   width=0.4, position=0, legend=False)
axes[1].set_ylim(0, metrics["Coverage"].max() * 1.30)
ax_nov.set_ylim(0, metrics["Novelty"].max() * 1.30)
axes[1].set_ylabel("coverage"); ax_nov.set_ylabel("novelty (bits)")
axes[1].set_title("Beyond-accuracy: coverage & novelty")
axes[1].set_xticklabels(order, rotation=35, ha="right")
axes[1].set_xlim(-0.6, len(order) - 0.4)
# Combined legend in the empty upper-left region
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color="#33617E", label="Coverage"),
                        Patch(color="#9A7B3F", label="Novelty")],
               loc="upper center", framealpha=0.95, fontsize=9)
fig.savefig(FIG_DIR / "metrics_comparison.png"); plt.close(fig)

print("Figures saved to", FIG_DIR)
for p in sorted(FIG_DIR.glob("*.png")):
    print(" -", p.name)
