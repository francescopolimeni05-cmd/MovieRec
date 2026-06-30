"""Data loading and preprocessing utilities.

Implements loading of MovieLens ratings/metadata, basic EDA statistics,
standard interaction filtering, optional user subsampling and a train/test
split (random or temporal).
"""

import numpy as np
import pandas as pd

from . import config


def load_ratings(path=config.RATINGS_PATH):
    """Load user-item ratings.

    Expected MovieLens columns: userId, movieId, rating, timestamp.
    Uses compact dtypes so the 32M-row file fits in ~0.6 GB of RAM.

    Returns:
        pandas.DataFrame
    """
    dtypes = {config.USER_COL: "int32", config.ITEM_COL: "int32",
              config.RATING_COL: "float32", config.TIMESTAMP_COL: "int64"}
    # Only request dtypes for columns that actually exist in the header.
    header = pd.read_csv(path, nrows=0).columns
    use_dtypes = {c: t for c, t in dtypes.items() if c in header}
    ratings = pd.read_csv(path, dtype=use_dtypes)
    required = {config.USER_COL, config.ITEM_COL, config.RATING_COL}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings file is missing required columns: {missing}")
    return ratings


def load_items(path=config.ITEMS_PATH):
    """Load item metadata.

    Expected MovieLens columns: movieId, title, genres

    Returns:
        pandas.DataFrame
    """
    items = pd.read_csv(path)
    required = {config.ITEM_COL, config.TITLE_COL, config.GENRES_COL}
    missing = required - set(items.columns)
    if missing:
        raise ValueError(f"items file is missing required columns: {missing}")
    return items


def describe_dataset(ratings, items=None, return_dict=False):
    """Compute and print basic dataset statistics used for the EDA section.

    Reports number of users/items/interactions, sparsity, rating distribution,
    most active users and most popular items.
    """
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_interactions = len(ratings)
    sparsity = 1.0 - n_interactions / (n_users * n_items)

    rating_dist = ratings[config.RATING_COL].value_counts().sort_index()
    most_active_users = ratings[config.USER_COL].value_counts().head(10)
    pop_counts = ratings[config.ITEM_COL].value_counts().head(10)

    most_popular = pop_counts.rename("n_ratings").to_frame()
    if items is not None:
        title_map = items.set_index(config.ITEM_COL)[config.TITLE_COL]
        most_popular[config.TITLE_COL] = most_popular.index.map(title_map)

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Users .............. {n_users:,}")
    print(f"Items (rated) ...... {n_items:,}")
    if items is not None:
        print(f"Items (catalog) .... {items[config.ITEM_COL].nunique():,}")
    print(f"Interactions ....... {n_interactions:,}")
    print(f"Sparsity ........... {sparsity:.4%}")
    print(f"Avg ratings/user ... {n_interactions / n_users:.1f}")
    print(f"Avg ratings/item ... {n_interactions / n_items:.1f}")
    print(f"Rating mean/median . {ratings[config.RATING_COL].mean():.2f} / "
          f"{ratings[config.RATING_COL].median():.2f}")
    print("\nRating distribution:")
    for r, c in rating_dist.items():
        print(f"  {r:>4}: {c:>8,}  ({c / n_interactions:5.1%})")
    print("\nTop-10 most popular items:")
    print(most_popular.to_string())
    print("=" * 60)

    if return_dict:
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_interactions,
            "sparsity": sparsity,
            "rating_distribution": rating_dist,
            "most_active_users": most_active_users,
            "most_popular_items": most_popular,
        }


def filter_dataset(ratings,
                   min_user_ratings=config.MIN_USER_RATINGS,
                   min_item_ratings=config.MIN_ITEM_RATINGS,
                   max_users=config.MAX_USERS,
                   max_items=getattr(config, "MAX_ITEMS", None),
                   random_state=config.RANDOM_STATE):
    """Apply standard interaction filtering + optional sub-sampling.

    Steps:
    1. Optionally cap the catalog to the `max_items` most popular items (keeps
       the item-item similarity matrix and MF tractable on 32M ratings).
    2. k-core pruning: iteratively drop users/items below the minimum support.
    3. Optionally keep a random subset of `max_users` users.
    Returns the filtered ratings DataFrame.
    """
    df = ratings
    if max_items is not None:
        top_items = df[config.ITEM_COL].value_counts().head(max_items).index
        df = df[df[config.ITEM_COL].isin(top_items)]

    while True:  # k-core pruning until both constraints hold
        before = len(df)
        item_counts = df[config.ITEM_COL].value_counts()
        keep_items = item_counts[item_counts >= min_item_ratings].index
        df = df[df[config.ITEM_COL].isin(keep_items)]
        user_counts = df[config.USER_COL].value_counts()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df[config.USER_COL].isin(keep_users)]
        if len(df) == before:
            break

    if max_users is not None:
        unique_users = df[config.USER_COL].unique()
        if len(unique_users) > max_users:
            rng = np.random.RandomState(random_state)
            sampled = rng.choice(unique_users, size=max_users, replace=False)
            df = df[df[config.USER_COL].isin(sampled)]
            # After dropping users, re-prune items that fell below support so the
            # CF/MF matrices stay dense enough to learn from.
            thr = getattr(config, "MIN_ITEM_RATINGS_AFTER_SAMPLE", 10)
            item_counts = df[config.ITEM_COL].value_counts()
            keep_items = item_counts[item_counts >= thr].index
            df = df[df[config.ITEM_COL].isin(keep_items)]

    return df.reset_index(drop=True)


def get_model_ratings(use_cache=True):
    """Return the filtered/sub-sampled modelling set, caching it to disk.

    The full 32M ratings file takes ~10s to read. We cache the much smaller
    filtered modelling set (a few hundred-k rows) to data/processed/ keyed by the
    relevant config parameters, so repeated runs are near-instant. Delete the
    cache or set use_cache=False to rebuild it.
    """
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    key = (f"u{config.MAX_USERS}_i{getattr(config, 'MAX_ITEMS', None)}"
           f"_mu{config.MIN_USER_RATINGS}_mi{config.MIN_ITEM_RATINGS}")
    cache = config.PROCESSED_DATA_DIR / f"model_ratings_{key}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    df = filter_dataset(load_ratings())
    df.to_pickle(cache)
    return df


def train_test_split_ratings(ratings, test_size=0.2,
                             random_state=config.RANDOM_STATE,
                             strategy="temporal"):
    """Create a per-user train/test split (leave-some-out).

    strategy="temporal" (default): the most recent `test_size` fraction of each
        user's interactions (by timestamp) goes to test. This avoids the
        train-on-future/test-on-past leakage of a random split and is the more
        realistic protocol.
    strategy="random": for each user a random `test_size` fraction of their
        ratings goes to test (kept for comparison / ablation).
    Both guarantee every test user is also present in train, which the
    personalized models require.
    """
    rng = np.random.RandomState(random_state)
    train_parts, test_parts = [], []

    for _, group in ratings.groupby(config.USER_COL, sort=False):
        n = len(group)
        n_test = int(round(n * test_size))
        if n_test == 0 or n_test >= n:
            train_parts.append(group)
            continue
        if strategy == "temporal" and config.TIMESTAMP_COL in group.columns:
            g = group.sort_values(config.TIMESTAMP_COL)
            test_parts.append(g.iloc[-n_test:])
            train_parts.append(g.iloc[:-n_test])
        else:
            idx = rng.permutation(n)
            mask = np.zeros(n, dtype=bool)
            mask[idx[:n_test]] = True
            test_parts.append(group.iloc[mask])
            train_parts.append(group.iloc[~mask])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def get_seen_items(ratings, user_id):
    """Return the set of items already rated/consumed by one user."""
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])
