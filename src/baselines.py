"""Non-personalized baseline recommenders.

Implements three baselines:
- MostPopularRecommender         (popularity by interaction count)
- HighestAverageRatingRecommender (quality with a minimum support)
- RandomRecommender              (lower bound / sanity check)

All recommenders share the same interface:
    fit(ratings, items=None) -> self
    recommend(user_id, ratings_train, n=10, exclude_seen=True) -> list[item_id]
"""

import numpy as np
import pandas as pd

from . import config


def _seen(ratings_train, user_id):
    return set(ratings_train.loc[ratings_train[config.USER_COL] == user_id,
                                 config.ITEM_COL])


class MostPopularRecommender:
    """Recommend the most frequently rated items (same list for every user)."""

    def __init__(self):
        self.ranking_ = None  # item ids ordered by popularity

    def fit(self, ratings, items=None):
        counts = ratings[config.ITEM_COL].value_counts()
        self.ranking_ = counts.index.to_numpy()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = _seen(ratings_train, user_id) if exclude_seen else set()
        recs = [i for i in self.ranking_ if i not in seen]
        return recs[:n]


class HighestAverageRatingRecommender:
    """Recommend items with the highest average rating and enough support."""

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking_ = None

    def fit(self, ratings, items=None):
        stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(
            ["mean", "count"])
        stats = stats[stats["count"] >= self.min_ratings]
        stats = stats.sort_values(["mean", "count"], ascending=False)
        self.ranking_ = stats.index.to_numpy()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = _seen(ratings_train, user_id) if exclude_seen else set()
        recs = [i for i in self.ranking_ if i not in seen]
        return recs[:n]


class BayesianAverageRecommender:
    """Damped/Bayesian weighted-rating baseline (IMDb formula, as taught).

    Instead of a hard minimum-ratings cut-off, items are scored with a
    shrinkage average that pulls items with few ratings toward the global mean:

        WR(j) = v/(v+m) * R_j  +  m/(v+m) * C

    where R_j = item j's mean rating, v = number of ratings for j,
    C = global mean rating across all items, m = smoothing weight
    (the prior strength, here the median rating count).
    """

    def __init__(self, m=None):
        self.m = m
        self.ranking_ = None

    def fit(self, ratings, items=None):
        stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(
            ["mean", "count"])
        C = ratings[config.RATING_COL].mean()
        m = self.m if self.m is not None else stats["count"].median()
        v, R = stats["count"], stats["mean"]
        stats["wr"] = (v / (v + m)) * R + (m / (v + m)) * C
        self.ranking_ = stats.sort_values("wr", ascending=False).index.to_numpy()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = _seen(ratings_train, user_id) if exclude_seen else set()
        recs = [i for i in self.ranking_ if i not in seen]
        return recs[:n]


class RandomRecommender:
    """Optional baseline: recommend random unseen items (lower bound)."""

    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        if items is not None:
            self.items_ = items[config.ITEM_COL].unique()
        else:
            self.items_ = ratings[config.ITEM_COL].unique()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = _seen(ratings_train, user_id) if exclude_seen else set()
        # Per-user deterministic shuffle for reproducibility.
        rng = np.random.RandomState(self.random_state + int(user_id))
        pool = [i for i in self.items_ if i not in seen]
        rng.shuffle(pool)
        return pool[:n]
