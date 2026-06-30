"""Collaborative filtering.

Item-item CF (main method) and user-user CF (extension), both based on
mean-centered ratings and cosine similarity with a top-k neighbourhood.

The user-item matrix is stored as a SciPy CSR matrix; ratings are centered by
the item mean (item-item) or the user mean (user-user) before computing cosine
similarity, which is the standard adjusted-cosine / Pearson-style formulation.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def _build_index(ids):
    ids = np.asarray(ids)
    return ids, {v: i for i, v in enumerate(ids)}


class ItemItemCollaborativeFiltering:
    """Item-item collaborative filtering using top-k cosine neighbourhoods."""

    def __init__(self, k=20, similarity="cosine"):
        self.k = k
        self.similarity = similarity
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.R_ = None              # centered user-item CSR (users x items)
        self.item_means_ = None
        self.sim_ = None            # dense item-item similarity (top-k kept)

    def fit(self, ratings):
        self.user_ids_, self.user_id_to_index_ = _build_index(
            ratings[config.USER_COL].unique())
        self.item_ids_, self.item_id_to_index_ = _build_index(
            ratings[config.ITEM_COL].unique())

        u = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        it = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        r = ratings[config.RATING_COL].to_numpy(dtype=float)

        n_users, n_items = len(self.user_ids_), len(self.item_ids_)

        # Item means for centering.
        self.item_means_ = np.zeros(n_items)
        np.add.at(self.item_means_, it, r)
        counts = np.zeros(n_items)
        np.add.at(counts, it, 1)
        counts[counts == 0] = 1
        self.item_means_ = self.item_means_ / counts

        centered = r - self.item_means_[it]
        self.R_ = csr_matrix((centered, (u, it)), shape=(n_users, n_items))

        # Item-item cosine similarity on centered columns.
        sim = cosine_similarity(self.R_.T, dense_output=True)
        np.fill_diagonal(sim, 0.0)

        # Keep only the top-k neighbours per item (zero the rest).
        if self.k is not None and self.k < n_items:
            for i in range(n_items):
                row = sim[i]
                if self.k < n_items:
                    cut = np.argpartition(row, -self.k)[-self.k:]
                    mask = np.ones(n_items, dtype=bool)
                    mask[cut] = False
                    row[mask] = 0.0
        # Store dense caches as float32 to halve memory (matters at scale and
        # for the ~1 GB deployment tier). The boolean "rated" mask is derived
        # per user from R_dense_ on the fly, so we do not store a second full
        # dense matrix. Speed is unchanged (per-user ops are single rows).
        self.sim_ = sim.astype(np.float32)
        self.abs_sim_ = np.abs(self.sim_)              # cache for normalization
        # Densify directly in float32 (avoids a transient float64 copy at scale).
        self.R_dense_ = self.R_.astype(np.float32).toarray()
        return self

    def _rated_row(self, uidx):
        return (self.R_dense_[uidx] != 0).astype(np.float32)

    def _user_scores(self, user_id):
        uidx = self.user_id_to_index_.get(user_id)
        if uidx is None:
            return None
        r_u = self.R_dense_[uidx]                       # centered ratings
        rated_mask = self._rated_row(uidx)
        numer = self.sim_.dot(r_u)
        denom = self.abs_sim_.dot(rated_mask)
        denom[denom == 0] = 1e-9
        scores = self.item_means_ + numer / denom
        return scores

    def predict_score(self, user_id, item_id):
        scores = self._user_scores(user_id)
        iidx = self.item_id_to_index_.get(item_id)
        if scores is None or iidx is None:
            return np.nan
        return float(scores[iidx])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        scores = self._user_scores(user_id)
        if scores is None:
            return []
        uidx = self.user_id_to_index_[user_id]
        rated_mask = self._rated_row(uidx)
        scores = scores.copy()
        if exclude_seen:
            scores[rated_mask.astype(bool)] = -np.inf
        # Items with no contributing neighbour get no signal.
        no_signal = self.abs_sim_.dot(rated_mask) == 0
        scores[no_signal] = -np.inf

        order = np.argsort(-scores)[:n]
        return [self.item_ids_[i] for i in order if np.isfinite(scores[i])]


class UserUserCollaborativeFiltering:
    """User-user collaborative filtering (extension)."""

    def __init__(self, k=30, similarity="cosine"):
        self.k = k
        self.similarity = similarity
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.R_ = None
        self.user_means_ = None
        self.sim_ = None

    def fit(self, ratings):
        self.user_ids_, self.user_id_to_index_ = _build_index(
            ratings[config.USER_COL].unique())
        self.item_ids_, self.item_id_to_index_ = _build_index(
            ratings[config.ITEM_COL].unique())

        u = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        it = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        r = ratings[config.RATING_COL].to_numpy(dtype=float)
        n_users, n_items = len(self.user_ids_), len(self.item_ids_)

        self.user_means_ = np.zeros(n_users)
        np.add.at(self.user_means_, u, r)
        counts = np.zeros(n_users)
        np.add.at(counts, u, 1)
        counts[counts == 0] = 1
        self.user_means_ = self.user_means_ / counts

        centered = r - self.user_means_[u]
        self.R_ = csr_matrix((centered, (u, it)), shape=(n_users, n_items))
        raw = csr_matrix((np.ones_like(r), (u, it)), shape=(n_users, n_items))
        self.rated_ = raw

        sim = cosine_similarity(self.R_, dense_output=True)
        np.fill_diagonal(sim, 0.0)
        if self.k is not None and self.k < n_users:
            for i in range(n_users):
                row = sim[i]
                cut = np.argpartition(row, -self.k)[-self.k:]
                mask = np.ones(n_users, dtype=bool)
                mask[cut] = False
                row[mask] = 0.0
        self.sim_ = sim.astype(np.float32)
        self.abs_sim_ = np.abs(self.sim_)
        self.R_dense_ = self.R_.astype(np.float32).toarray()       # centered (users x items)
        self.rated_dense_ = self.rated_.astype(np.float32).toarray()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        uidx = self.user_id_to_index_.get(user_id)
        if uidx is None:
            return []
        sim_u = self.sim_[uidx]                         # similarity to all users
        abs_sim_u = self.abs_sim_[uidx]
        numer = sim_u.dot(self.R_dense_)                # weighted centered ratings
        contrib = abs_sim_u.dot(self.rated_dense_)
        denom = np.where(contrib == 0, 1e-9, contrib)
        scores = self.user_means_[uidx] + numer / denom

        scores[self.rated_dense_[uidx].astype(bool)] = -np.inf if exclude_seen else 0
        scores[contrib == 0] = -np.inf

        order = np.argsort(-scores)[:n]
        return [self.item_ids_[i] for i in order if np.isfinite(scores[i])]
