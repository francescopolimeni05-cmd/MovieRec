"""Hyper-parameter tuning experiment (a 'stronger grade' extension).

Sweeps the neighbourhood size k for item-item CF and the number of latent
factors for the Truncated-SVD matrix factorization, scoring each setting with
NDCG@10 on a held-out user sample. Saves results/tuning.csv and a plot.

Run:  python tuning.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_loading import get_model_ratings, train_test_split_ratings
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model

K = config.TOP_K
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(config.RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

ratings = get_model_ratings(use_cache=True)
train, test = train_test_split_ratings(ratings, test_size=0.2)
pop = train[config.ITEM_COL].value_counts().to_dict()
n_users = train[config.USER_COL].nunique()
catalog = train[config.ITEM_COL].unique()

rng = np.random.RandomState(config.RANDOM_STATE)
users = rng.choice(test[config.USER_COL].unique(),
                   size=min(300, test[config.USER_COL].nunique()), replace=False)


def ndcg_of(model):
    res = evaluate_model(model, train, test, users, k=K, all_items=catalog,
                         item_popularity=pop, n_total_users=n_users)
    return res[f"NDCG@{K}"], res.get("Coverage", np.nan)


rows = []

print(">> Tuning item-item CF neighbourhood size k ...")
for k in [10, 20, 30, 50, 80]:
    ndcg, cov = ndcg_of(ItemItemCollaborativeFiltering(k=k).fit(train))
    rows.append({"model": "ItemItemCF", "param": "k", "value": k,
                 "NDCG@10": ndcg, "Coverage": cov})
    print(f"   k={k:<3} NDCG@10={ndcg:.4f}  Coverage={cov:.3f}")

print(">> Tuning Truncated-SVD number of latent factors ...")
for f in [10, 20, 50, 100, 150]:
    ndcg, cov = ndcg_of(MatrixFactorizationRecommender(n_factors=f).fit(train))
    rows.append({"model": "MatrixFactorization", "param": "n_factors", "value": f,
                 "NDCG@10": ndcg, "Coverage": cov})
    print(f"   n_factors={f:<4} NDCG@10={ndcg:.4f}  Coverage={cov:.3f}")

df = pd.DataFrame(rows)
out = config.RESULTS_DIR / "tuning.csv"
df.to_csv(out, index=False)
print(f"\nSaved -> {out}")

# Plot the two sweeps side by side.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (mdl, xlabel) in zip(axes, [("ItemItemCF", "neighbourhood size k"),
                                    ("MatrixFactorization", "latent factors")]):
    sub = df[df.model == mdl]
    ax.plot(sub["value"], sub["NDCG@10"], "o-", color="#C44E52")
    ax.set_title(mdl); ax.set_xlabel(xlabel); ax.set_ylabel("NDCG@10")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "figures" / "tuning.png", dpi=120)
print("Saved -> results/figures/tuning.png")
