"""Evaluation metrics for recommender systems.

Top-N ranking metrics (Precision@K, Recall@K, Hit-Rate@K, NDCG@K, MRR) plus
beyond-accuracy metrics (catalog coverage, novelty, intra-list diversity) and a
driver function `evaluate_model` that averages metrics over a set of users.
Optional rating-prediction metrics (MAE/RMSE) are included as well.
"""

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Per-user ranking metrics (binary relevance)
# --------------------------------------------------------------------------- #
def precision_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return np.nan
    rec_k = list(recommended_items)[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for i in rec_k if i in relevant_items)
    return hits / len(rec_k)


def recall_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return np.nan
    rec_k = list(recommended_items)[:k]
    hits = sum(1 for i in rec_k if i in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return np.nan
    rec_k = list(recommended_items)[:k]
    return 1.0 if any(i in relevant_items for i in rec_k) else 0.0


def dcg_at_k(relevance_scores, k=10):
    rel = np.asarray(relevance_scores, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))  # ranks start at 1
    return float(np.sum(rel / discounts))


def ndcg_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return np.nan
    rec_k = list(recommended_items)[:k]
    rel = [1.0 if i in relevant_items else 0.0 for i in rec_k]
    dcg = dcg_at_k(rel, k)
    ideal = dcg_at_k([1.0] * min(len(relevant_items), k), k)
    return dcg / ideal if ideal > 0 else 0.0


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    rec_k = list(recommended_items)[:k]
    for rank, item in enumerate(rec_k, start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# Beyond-accuracy metrics (computed over all users' recommendation lists)
# --------------------------------------------------------------------------- #
def catalog_coverage(all_recommendations, all_items):
    """Fraction of the catalog that appears in at least one recommendation."""
    recommended = set()
    for recs in all_recommendations:
        recommended.update(recs)
    catalog = set(all_items)
    if not catalog:
        return 0.0
    return len(recommended & catalog) / len(catalog)


def novelty(all_recommendations, item_popularity, n_users):
    """Mean self-information (-log2 popularity) of recommended items.

    item_popularity: dict item_id -> number of users who interacted with it.
    Higher = more novel (less popular) recommendations.
    """
    vals = []
    for recs in all_recommendations:
        for item in recs:
            p = item_popularity.get(item, 1) / max(n_users, 1)
            p = min(max(p, 1e-9), 1.0)
            vals.append(-np.log2(p))
    return float(np.mean(vals)) if vals else np.nan


def intra_list_diversity(all_recommendations, item_vectors, item_index):
    """Average (1 - cosine similarity) between item pairs within each list."""
    from sklearn.metrics.pairwise import cosine_similarity
    diversities = []
    for recs in all_recommendations:
        idxs = [item_index[i] for i in recs if i in item_index]
        if len(idxs) < 2:
            continue
        sub = item_vectors[idxs]
        sims = cosine_similarity(sub)
        n = sims.shape[0]
        upper = sims[np.triu_indices(n, k=1)]
        diversities.append(1.0 - float(np.mean(upper)))
    return float(np.mean(diversities)) if diversities else np.nan


# --------------------------------------------------------------------------- #
# Rating-prediction metrics (optional)
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_rating_prediction(model, ratings_test, max_users=400,
                               random_state=config.RANDOM_STATE):
    """MAE/RMSE of a model's predicted ratings on held-out test ratings.

    Grouped by user for efficiency: item-item CF reuses its per-user score
    vector (`_user_scores`), MF predicts via the latent dot product
    (`predict_score`). Models without a rating-prediction interface (e.g.
    user-user CF here) are skipped. Evaluated on a sample of test users.
    """
    has_user_scores = hasattr(model, "_user_scores")
    if not has_user_scores and not hasattr(model, "predict_score"):
        return {"MAE": np.nan, "RMSE": np.nan, "n_pairs": 0}

    users = ratings_test[config.USER_COL].unique()
    if len(users) > max_users:
        rng = np.random.RandomState(random_state)
        users = rng.choice(users, size=max_users, replace=False)
    test_by_user = {u: g for u, g in ratings_test.groupby(config.USER_COL)}

    y_true, y_pred = [], []
    for u in users:
        g = test_by_user.get(u)
        if g is None:
            continue
        if has_user_scores:
            sv = model._user_scores(u)
            if sv is None:
                continue
            idx_map = model.item_id_to_index_
            for it, r in zip(g[config.ITEM_COL], g[config.RATING_COL]):
                j = idx_map.get(it)
                if j is not None and np.isfinite(sv[j]):
                    y_true.append(r); y_pred.append(sv[j])
        else:
            for it, r in zip(g[config.ITEM_COL], g[config.RATING_COL]):
                p = model.predict_score(u, it)
                if p is not None and np.isfinite(p):
                    y_true.append(r); y_pred.append(p)
    if not y_true:
        return {"MAE": np.nan, "RMSE": np.nan, "n_pairs": 0}
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "n_pairs": len(y_true)}


# --------------------------------------------------------------------------- #
# Driver: evaluate a model over a set of users
# --------------------------------------------------------------------------- #
def evaluate_model(model, ratings_train, ratings_test, users, k=10,
                   relevance_threshold=config.RELEVANCE_THRESHOLD,
                   all_items=None, item_popularity=None, n_total_users=None,
                   return_recs=False):
    """Average top-K metrics over `users`.

    A test item is "relevant" for a user if its held-out rating is
    >= relevance_threshold. Users with no relevant test item are skipped.
    """
    # Pre-group test relevant items per user.
    test_pos = ratings_test[ratings_test[config.RATING_COL] >= relevance_threshold]
    relevant_by_user = (test_pos.groupby(config.USER_COL)[config.ITEM_COL]
                        .apply(set).to_dict())

    per_user = {"precision": [], "recall": [], "hit_rate": [],
                "ndcg": [], "mrr": []}
    all_recs = []

    for user_id in users:
        relevant = relevant_by_user.get(user_id)
        if not relevant:
            continue
        recs = model.recommend(user_id, ratings_train, n=k, exclude_seen=True)
        all_recs.append(recs)
        per_user["precision"].append(precision_at_k(recs, relevant, k))
        per_user["recall"].append(recall_at_k(recs, relevant, k))
        per_user["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
        per_user["ndcg"].append(ndcg_at_k(recs, relevant, k))
        per_user["mrr"].append(mean_reciprocal_rank(recs, relevant, k))

    results = {f"{m}@{k}" if m in ("precision", "recall", "hit_rate", "ndcg")
               else m: np.nanmean(v) if v else np.nan
               for m, v in per_user.items()}
    # Friendlier metric names.
    results = {
        f"Precision@{k}": np.nanmean(per_user["precision"]) if per_user["precision"] else np.nan,
        f"Recall@{k}": np.nanmean(per_user["recall"]) if per_user["recall"] else np.nan,
        f"HitRate@{k}": np.nanmean(per_user["hit_rate"]) if per_user["hit_rate"] else np.nan,
        f"NDCG@{k}": np.nanmean(per_user["ndcg"]) if per_user["ndcg"] else np.nan,
        "MRR": np.nanmean(per_user["mrr"]) if per_user["mrr"] else np.nan,
        "n_users_eval": len(all_recs),
    }
    if all_items is not None:
        results["Coverage"] = catalog_coverage(all_recs, all_items)
    if item_popularity is not None and n_total_users is not None:
        results["Novelty"] = novelty(all_recs, item_popularity, n_total_users)

    # 95% confidence interval (half-width) of NDCG@k via the standard error of
    # the mean over users: 1.96 * std / sqrt(n). Lets us report mean ± CI.
    ndcg_vals = np.asarray(per_user["ndcg"], dtype=float)
    ndcg_vals = ndcg_vals[~np.isnan(ndcg_vals)]
    if ndcg_vals.size > 1:
        results[f"NDCG@{k}_ci95"] = float(
            1.96 * ndcg_vals.std(ddof=1) / np.sqrt(ndcg_vals.size))
    else:
        results[f"NDCG@{k}_ci95"] = np.nan

    if return_recs:
        return results, all_recs
    return results
