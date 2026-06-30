"""Cold-start preference-elicitation recommender for brand-new users.

A new user has no ratings, so neither CF nor matrix factorization can help them
(the cold-start problem flagged in the report). Instead we ask a few questions
and build a **content profile** from the answers: a weighted vector over movie
genres. Movies are then ranked by cosine similarity to that profile, adjusted by
the user's stated preference for popular vs. niche titles and filtered by era and
quality. This is the standard onboarding approach used by real services.
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from . import config

# A "mood" maps to a handful of genres, so the questionnaire can stay friendly.
MOOD_GENRES = {
    "Light & fun": ["Comedy", "Animation", "Children", "Adventure"],
    "Intense & gripping": ["Thriller", "Crime", "Action", "Mystery"],
    "Thought-provoking": ["Drama", "Documentary", "War", "Film-Noir"],
    "Heartwarming": ["Romance", "Family", "Drama", "Musical"],
    "Scary": ["Horror", "Thriller", "Mystery"],
    "Epic & adventurous": ["Adventure", "Fantasy", "Sci-Fi", "Action"],
}

# Curated, friendly genre choices for the questionnaire.
GENRE_CHOICES = ["Action", "Adventure", "Animation", "Comedy", "Crime",
                 "Documentary", "Drama", "Fantasy", "Horror", "Musical",
                 "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

ERA_CHOICES = ["Any era", "Classics (pre-1980)", "80s–90s", "2000s", "Recent (2010+)"]
POP_CHOICES = ["Popular blockbusters", "A balanced mix", "Hidden gems"]
QUALITY_CHOICES = ["Any", "Only well-rated films"]


def _parse_year(title):
    m = re.search(r"\((\d{4})\)", str(title))
    return int(m.group(1)) if m else None


def _in_era(year, era):
    if year is None or era == "Any era":
        return era == "Any era"
    if era == "Classics (pre-1980)":
        return year < 1980
    if era == "80s–90s":
        return 1980 <= year <= 1999
    if era == "2000s":
        return 2000 <= year <= 2009
    if era == "Recent (2010+)":
        return year >= 2010
    return True


class NewUserProfiler:
    """Build a taste profile from questionnaire answers and recommend movies."""

    def __init__(self, min_support=20):
        self.min_support = min_support

    def fit(self, ratings, items):
        items = items.drop_duplicates(subset=config.ITEM_COL).reset_index(drop=True)
        genres_split = (items[config.GENRES_COL].fillna("")
                        .str.replace("Children's", "Children", regex=False)
                        .str.split("|"))

        vocab = sorted({g for gs in genres_split for g in gs
                        if g and g != "(no genres listed)"})
        self.vocab = vocab
        self.gindex = {g: i for i, g in enumerate(vocab)}

        M = np.zeros((len(items), len(vocab)), dtype=float)
        for r, gs in enumerate(genres_split):
            for g in gs:
                j = self.gindex.get(g)
                if j is not None:
                    M[r, j] = 1.0
        self.item_ids = items[config.ITEM_COL].to_numpy()
        self.item_norm = normalize(M)  # row-normalized -> dot product = cosine

        self.title = items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()
        self.genres = items.set_index(config.ITEM_COL)[config.GENRES_COL].to_dict()
        self.year = {iid: _parse_year(t) for iid, t in self.title.items()}

        # Popularity + quality (damped/Bayesian rating) from the rating data.
        stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(["mean", "count"])
        C = ratings[config.RATING_COL].mean()
        m = stats["count"].median()
        stats["wr"] = ((stats["count"] / (stats["count"] + m)) * stats["mean"]
                       + (m / (stats["count"] + m)) * C)
        self.support = stats["count"].to_dict()
        self.avg = stats["mean"].to_dict()
        self.quality = stats["wr"].to_dict()
        self.max_pop = float(stats["count"].max()) if len(stats) else 1.0
        return self

    def build_profile(self, genres, mood=None):
        """Weighted genre vector: chosen genres count double, mood genres once."""
        prof = np.zeros(len(self.vocab))
        for g in (genres or []):
            j = self.gindex.get(g)
            if j is not None:
                prof[j] += 2.0
        for g in MOOD_GENRES.get(mood, []):
            j = self.gindex.get(g)
            if j is not None:
                prof[j] += 1.0
        if prof.sum() == 0:          # no preference -> flat (popularity will rank)
            prof[:] = 1.0
        norm = np.linalg.norm(prof)
        return prof / norm if norm else prof

    def recommend(self, genres=None, mood=None, era="Any era",
                  popularity="A balanced mix", quality="Any", n=10):
        profile = self.build_profile(genres, mood)
        scores = self.item_norm @ profile          # cosine similarity, in [0, 1]

        pop = np.array([self.support.get(i, 0) / self.max_pop for i in self.item_ids])
        if popularity == "Popular blockbusters":
            scores = scores + 0.35 * pop
        elif popularity == "Hidden gems":
            scores = scores - 0.35 * pop
        # A balanced mix -> leave scores as-is.

        # Filters (support, quality, era) via a validity mask.
        mask = np.ones(len(self.item_ids), dtype=bool)
        for idx, iid in enumerate(self.item_ids):
            if self.support.get(iid, 0) < self.min_support:
                mask[idx] = False
                continue
            if quality == "Only well-rated films" and self.quality.get(iid, 0) < 3.6:
                mask[idx] = False
                continue
            if not _in_era(self.year.get(iid), era):
                mask[idx] = False
        scores = np.where(mask, scores, -np.inf)

        order = np.argsort(-scores)[:n]
        out = []
        for i in order:
            if not np.isfinite(scores[i]):
                continue
            iid = self.item_ids[i]
            out.append({
                "movieId": iid,
                "title": self.title.get(iid, str(iid)),
                "genres": self.genres.get(iid, ""),
                "year": self.year.get(iid),
                "avg_rating": round(float(self.avg.get(iid, np.nan)), 2),
                "match": round(float(self.item_norm[i] @ profile), 3),
            })
        return out
