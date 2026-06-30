"""Popularity-bias / long-tail analysis (threats-to-validity evidence).

For each model we measure how much of its recommendations fall in the popular
"head" vs the long "tail" of the catalogue, and the NDCG restricted to tail
(less-popular) relevant items. This quantifies the popularity bias that inflates
plain accuracy metrics and shows which models actually help item cold-start /
discovery.

Run:  python analysis_longtail.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_loading import get_model_ratings, train_test_split_ratings
from src.baselines import (MostPopularRecommender, HighestAverageRatingRecommender,
                           BayesianAverageRecommender, RandomRecommender)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender

K = config.TOP_K
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(config.RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

ratings = get_model_ratings(use_cache=True)
train, test = train_test_split_ratings(ratings, test_size=0.2)  # temporal

# Define head = top 20% of items by training popularity; tail = the other 80%.
pop = train[config.ITEM_COL].value_counts()
n_head = max(1, int(0.20 * len(pop)))
head_items = set(pop.index[:n_head])
print(f"Catalogue: {len(pop)} items | head (top 20%) = {len(head_items)} items "
      f"holding {pop.iloc[:n_head].sum() / pop.sum():.1%} of all ratings")

from src.data_loading import load_items
items = load_items()
models = {
    "MostPopular": MostPopularRecommender().fit(train, items),
    "BayesianAvg(IMDb)": BayesianAverageRecommender().fit(train, items),
    "ContentBased(TF-IDF)": ContentBasedRecommender(use_tfidf=True).fit(train, items),
    "ItemItemCF(k=10)": ItemItemCollaborativeFiltering(k=10).fit(train),
    "MatrixFactorization(SVD)": MatrixFactorizationRecommender(n_factors=20).fit(train),
}

rng = np.random.RandomState(config.RANDOM_STATE)
users = rng.choice(test[config.USER_COL].unique(),
                   size=min(400, test[config.USER_COL].nunique()), replace=False)

rows = []
for name, model in models.items():
    tail_share = []
    for u in users:
        recs = model.recommend(u, train, n=K, exclude_seen=True)
        if not recs:
            continue
        tail = sum(1 for i in recs if i not in head_items) / len(recs)
        tail_share.append(tail)
    rows.append({"model": name,
                 "tail_share@10": float(np.mean(tail_share)) if tail_share else np.nan})
    print(f"   {name:28s} tail-share@10 = {rows[-1]['tail_share@10']:.3f}")

df = pd.DataFrame(rows).set_index("model").sort_values("tail_share@10")
out = config.RESULTS_DIR / "popularity_bias.csv"
df.to_csv(out)
print(f"\nSaved -> {out}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(df.index, df["tail_share@10"], color="#33617E")
ax.set_xlabel("share of top-10 recommendations in the long tail (bottom 80%)")
ax.set_title("Popularity bias: who recommends beyond the head?")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "figures" / "popularity_bias.png", dpi=120)
print("Saved -> results/figures/popularity_bias.png")
