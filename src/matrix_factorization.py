"""Matrix factorization via Truncated SVD — the method taught in the course.

The course notebooks (mf.ipynb, mf_movielens.ipynb, mf_books_student.ipynb)
factorize the sparse user-item matrix with sklearn's TruncatedSVD:

    R ≈ U · Vᵀ                      (low-rank approximation, R ≈ P·Qᵀ)
    user_factors = svd.fit_transform(R)     # U  (n_users × k)
    item_factors = svd.components_.T        # V  (n_items × k)
    prediction(u, i) = user_factors[u] · item_factors[i]      (dot product)
    top-N = argsort of the predicted scores for unseen items

We follow exactly that recipe: TruncatedSVD on a scipy.sparse matrix,
predictions via dot product, recommendations via argsort.

Methodological note (see REPORT "Threats to validity"): factorizing the raw
matrix treats every *missing* entry as a 0, i.e. as an implicit "dislike"
(imputation-by-zero). This is the course's pedagogical choice and works well for
ranking, but it makes the model lean toward popular items and is less principled
for explicit feedback than an MF trained only on observed ratings (mean-centered
SVD or biased SGD). We keep the course method and document the trade-off.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from . import config


class MatrixFactorizationRecommender:
    """Truncated-SVD matrix factorization (course method)."""

    def __init__(self, n_factors=50, random_state=config.RANDOM_STATE):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd_ = None
        self.user_factors_ = None     # U  (n_users × k)
        self.item_factors_ = None     # V  (n_items × k)
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.train_seen_ = None       # uidx -> set(iidx)

    def fit(self, ratings):
        self.user_ids_ = ratings[config.USER_COL].unique()
        self.item_ids_ = ratings[config.ITEM_COL].unique()
        self.user_id_to_index_ = {v: i for i, v in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {v: i for i, v in enumerate(self.item_ids_)}

        u = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        i = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        r = ratings[config.RATING_COL].to_numpy(dtype=float)
        n_users, n_items = len(self.user_ids_), len(self.item_ids_)

        # Sparse user-item rating matrix (missing entries stay 0, as in the course).
        R = csr_matrix((r, (u, i)), shape=(n_users, n_items))

        # TruncatedSVD requires n_components < n_features.
        k = min(self.n_factors, min(R.shape) - 1)
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.user_factors_ = self.svd_.fit_transform(R)      # U
        self.item_factors_ = self.svd_.components_.T          # V (items × k)

        self.train_seen_ = {}
        for uu, ii in zip(u, i):
            self.train_seen_.setdefault(uu, set()).add(ii)
        return self

    def predict_score(self, user_id, item_id):
        uidx = self.user_id_to_index_.get(user_id)
        iidx = self.item_id_to_index_.get(item_id)
        if uidx is None or iidx is None:
            return np.nan
        return float(self.user_factors_[uidx] @ self.item_factors_[iidx])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        uidx = self.user_id_to_index_.get(user_id)
        if uidx is None:
            return []
        # Predicted score for every item = dot product (course's approach).
        scores = self.item_factors_ @ self.user_factors_[uidx]
        if exclude_seen:
            seen = self.train_seen_.get(uidx, set())
            if seen:
                scores = scores.copy()
                scores[list(seen)] = -np.inf
        order = np.argsort(-scores)[:n]
        return [self.item_ids_[i] for i in order if np.isfinite(scores[i])]
