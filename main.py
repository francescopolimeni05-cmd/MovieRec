"""End-to-end pipeline for the individual recommender-systems project.

Steps
-----
1. Load MovieLens ratings + movie metadata.
2. EDA on the full dataset.
3. Filtering + (optional) user subsampling for fast modelling.
4. Per-user train/test split.
5. Train baselines, content-based, item-item CF, user-user CF, matrix factorization.
6. Evaluate all models with the same protocol (Precision/Recall/NDCG/MRR + coverage/novelty).
7. Save the comparison table to results/metrics.csv and qualitative example
   recommendations for 3 users to results/example_recommendations.csv.

Run:  python main.py
"""

import os
import time
import numpy as np
import pandas as pd

from src import config
from src.data_loading import (load_ratings, load_items, describe_dataset,
                              filter_dataset, train_test_split_ratings,
                              get_model_ratings)
from src.baselines import (MostPopularRecommender, HighestAverageRatingRecommender,
                           BayesianAverageRecommender, RandomRecommender)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import (ItemItemCollaborativeFiltering,
                                         UserUserCollaborativeFiltering)
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model, evaluate_rating_prediction

K = config.TOP_K
# Full-data EDA reads the 877 MB ratings file (~10s). Set RUN_FULL_EDA=0 to skip
# it on fast re-runs (EDA is also available in the notebook and generate_eda.py).
RUN_FULL_EDA = os.environ.get("RUN_FULL_EDA", "1") != "0"


def main():
    t0 = time.time()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load ------------------------------------------------------- #
    print(">> Loading data ...")
    items = load_items()

    # ----- 2. EDA on full data (skippable for fast re-runs) ------------- #
    if RUN_FULL_EDA:
        describe_dataset(load_ratings(), items)

    # ----- 3. Filter + subsample (cached to data/processed/) ------------ #
    print("\n>> Filtering / subsampling for modelling ...")
    model_ratings = get_model_ratings(use_cache=True)
    print(f"   After filtering: {len(model_ratings):,} ratings | "
          f"{model_ratings[config.USER_COL].nunique():,} users | "
          f"{model_ratings[config.ITEM_COL].nunique():,} items")

    # ----- 4. Train/test split ------------------------------------------ #
    train, test = train_test_split_ratings(model_ratings, test_size=0.2,
                                            strategy="temporal")
    print(f"   Train: {len(train):,}  |  Test: {len(test):,}")

    # Popularity for the novelty metric (training set).
    item_pop = train[config.ITEM_COL].value_counts().to_dict()
    n_total_users = train[config.USER_COL].nunique()
    catalog_items = train[config.ITEM_COL].unique()

    # ----- 5. Train models ---------------------------------------------- #
    print("\n>> Training models ...")
    models = {}

    def timed(name, builder):
        s = time.time()
        m = builder()
        print(f"   [{name}] trained in {time.time() - s:.1f}s")
        models[name] = m

    timed("MostPopular", lambda: MostPopularRecommender().fit(train, items))
    timed("HighestAvg(min20)",
          lambda: HighestAverageRatingRecommender(min_ratings=20).fit(train, items))
    timed("BayesianAvg(IMDb)",
          lambda: BayesianAverageRecommender().fit(train, items))
    timed("Random", lambda: RandomRecommender().fit(train, items))
    timed("ContentBased(TF-IDF)",
          lambda: ContentBasedRecommender(use_tfidf=True).fit(train, items))
    timed("ContentBased(raw genres)",
          lambda: ContentBasedRecommender(use_tfidf=False).fit(train, items))
    # k and n_factors selected via tuning.py (see results/tuning.csv).
    timed("ItemItemCF(k=10)",
          lambda: ItemItemCollaborativeFiltering(k=10).fit(train))
    timed("UserUserCF(k=40)",
          lambda: UserUserCollaborativeFiltering(k=40).fit(train))
    timed("MatrixFactorization(TruncatedSVD)",
          lambda: MatrixFactorizationRecommender(n_factors=20).fit(train))

    # ----- 6. Evaluate --------------------------------------------------- #
    print("\n>> Evaluating (this is the slowest part) ...")
    # Evaluate on a reproducible sample of test users to keep runtime low.
    all_test_users = test[config.USER_COL].unique()
    EVAL_SAMPLE = 350
    if len(all_test_users) > EVAL_SAMPLE:
        rng = np.random.RandomState(config.RANDOM_STATE)
        eval_users = rng.choice(all_test_users, size=EVAL_SAMPLE, replace=False)
    else:
        eval_users = all_test_users
    print(f"   Evaluating on {len(eval_users)} users")
    rows = []
    for name, model in models.items():
        s = time.time()
        res = evaluate_model(model, train, test, eval_users, k=K,
                             all_items=catalog_items,
                             item_popularity=item_pop,
                             n_total_users=n_total_users)
        res["model"] = name
        rows.append(res)
        print(f"   [{name}] P@{K}={res[f'Precision@{K}']:.4f} "
              f"R@{K}={res[f'Recall@{K}']:.4f} "
              f"NDCG@{K}={res[f'NDCG@{K}']:.4f} "
              f"Cov={res.get('Coverage', float('nan')):.3f} "
              f"({time.time() - s:.1f}s)")

    metrics = pd.DataFrame(rows).set_index("model")
    cols = [f"Precision@{K}", f"Recall@{K}", f"HitRate@{K}", f"NDCG@{K}",
            f"NDCG@{K}_ci95", "MRR", "Coverage", "Novelty", "n_users_eval"]
    metrics = metrics[[c for c in cols if c in metrics.columns]]
    metrics = metrics.sort_values(f"NDCG@{K}", ascending=False)

    out_path = config.RESULTS_DIR / "metrics.csv"
    metrics.to_csv(out_path)
    print("\n>> Comparison table (sorted by NDCG):")
    print(metrics.round(4).to_string())
    print(f"\n   Saved -> {out_path}")

    # ----- 6b. Rating-prediction metrics (MAE / RMSE) ------------------- #
    # Item-item CF predicts on the actual rating scale (r_i + weighted dev), so
    # MAE/RMSE are meaningful. TruncatedSVD reconstructs the raw matrix (not
    # rating-calibrated), so it is used for ranking, not rating prediction.
    print("\n>> Rating prediction (MAE / RMSE) — item-item CF ...")
    rp_rows = []
    for name in ["ItemItemCF(k=10)"]:
        rp = evaluate_rating_prediction(models[name], test)
        rp["model"] = name
        rp_rows.append(rp)
        print(f"   [{name}] MAE={rp['MAE']:.3f} RMSE={rp['RMSE']:.3f} "
              f"(n={rp['n_pairs']:,})")
    rp_df = pd.DataFrame(rp_rows).set_index("model")[["MAE", "RMSE", "n_pairs"]]
    rp_path = config.RESULTS_DIR / "rating_prediction.csv"
    rp_df.to_csv(rp_path)
    print(f"   Saved -> {rp_path}")

    # ----- 7. Example recommendations for 3 users ----------------------- #
    print("\n>> Example recommendations for 3 users ...")
    title_map = items.set_index(config.ITEM_COL)[config.TITLE_COL]
    example_users = list(eval_users[:3])
    example_rows = []
    for uid in example_users:
        for name, model in models.items():
            recs = model.recommend(uid, train, n=5, exclude_seen=True)
            titles = [title_map.get(i, str(i)) for i in recs]
            example_rows.append({
                "user": uid, "model": name,
                "top5_titles": " | ".join(titles)})
    examples = pd.DataFrame(example_rows)
    ex_path = config.RESULTS_DIR / "example_recommendations.csv"
    examples.to_csv(ex_path, index=False)
    print(f"   Saved -> {ex_path}")

    print(f"\nDone in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
