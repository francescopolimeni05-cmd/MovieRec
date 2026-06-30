"""Content-based recommender.

Item feature vectors are built from MovieLens genres (TF-IDF over the
genre "documents"). A user profile is the rating-weighted, mean-centered sum
of the vectors of the items the user has rated. Recommendations are the unseen
items whose vectors are most cosine-similar to the user profile.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from . import config


class ContentBasedRecommender:
    """Content-based recommender using item metadata (genres)."""

    def __init__(self, feature_col=config.GENRES_COL, use_tfidf=True):
        self.feature_col = feature_col
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.item_features_ = None        # (n_items, n_features) sparse/dense
        self.item_ids_ = None             # array of item ids (row order)
        self.item_id_to_index_ = None

    def _to_text(self, series):
        # MovieLens genres look like "Action|Adventure|Sci-Fi".
        return (series.fillna("")
                .str.replace("|", " ", regex=False)
                .str.replace("-", "", regex=False)
                .str.lower())

    def fit(self, ratings, items):
        items = items.drop_duplicates(subset=config.ITEM_COL)
        text = self._to_text(items[self.feature_col])

        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            self.vectorizer = CountVectorizer(token_pattern=r"[^\s]+")

        feats = self.vectorizer.fit_transform(text)
        # L2-normalize rows so dot product == cosine similarity.
        self.item_features_ = normalize(feats)
        self.item_ids_ = items[config.ITEM_COL].to_numpy()
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.item_ids_)}
        return self

    def build_user_profile(self, user_id, ratings_train):
        """profile(u) = sum_i (r(u,i) - mean_r(u)) * vec(i)."""
        ur = ratings_train[ratings_train[config.USER_COL] == user_id]
        if ur.empty:
            return None
        mean_r = ur[config.RATING_COL].mean()

        rows, weights = [], []
        for item, r in zip(ur[config.ITEM_COL], ur[config.RATING_COL]):
            idx = self.item_id_to_index_.get(item)
            if idx is not None:
                rows.append(idx)
                weights.append(r - mean_r)
        if not rows:
            return None

        weights = np.asarray(weights)
        # Weighted sum of (sparse) item vectors -> dense profile vector.
        profile = self.item_features_[rows].T.dot(weights)
        profile = np.asarray(profile).ravel()
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        return profile

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None or not np.any(profile):
            return []
        scores = self.item_features_.dot(profile)  # cosine (rows normalized)
        scores = np.asarray(scores).ravel()

        seen = set()
        if exclude_seen:
            seen = set(ratings_train.loc[
                ratings_train[config.USER_COL] == user_id, config.ITEM_COL])

        order = np.argsort(-scores)
        recs = []
        for idx in order:
            item = self.item_ids_[idx]
            if item in seen:
                continue
            recs.append(item)
            if len(recs) >= n:
                break
        return recs

    def similar_items(self, item_id, n=10):
        """Return the n most genre-similar items to a given item."""
        idx = self.item_id_to_index_.get(item_id)
        if idx is None:
            return []
        sims = cosine_similarity(self.item_features_[idx],
                                 self.item_features_).ravel()
        order = np.argsort(-sims)
        return [self.item_ids_[i] for i in order if i != idx][:n]
