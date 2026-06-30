"""Interactive Streamlit demo for the movie recommender prototype.

Run from the project root:

    streamlit run app.py

Two modes (tabs):
1. **Build your profile** — a new user answers a short questionnaire; we build a
   content/taste profile from the answers and recommend the best-matching movies
   (a cold-start / preference-elicitation onboarding).
2. **Explore by user** — pick an existing MovieLens user and compare the top-N
   recommendations of every algorithm, inspect their taste profile, and find
   movies similar to a chosen title.

Models are trained once and cached.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data_loading import get_model_ratings, load_items, train_test_split_ratings
from src.baselines import (MostPopularRecommender, HighestAverageRatingRecommender,
                           BayesianAverageRecommender)
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.cold_start import (NewUserProfiler, GENRE_CHOICES, MOOD_GENRES,
                            ERA_CHOICES, POP_CHOICES, QUALITY_CHOICES)

st.set_page_config(page_title="Movie Recommender Prototype", page_icon="🎬",
                   layout="wide")


# --------------------------------------------------------------------------- #
# Data + models (trained once, cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading data and training models…")
def load_everything():
    ratings = get_model_ratings(use_cache=True)
    items = load_items()
    train, _ = train_test_split_ratings(ratings, test_size=0.2)

    models = {
        "Most Popular": MostPopularRecommender().fit(train, items),
        "Highest Average": HighestAverageRatingRecommender(min_ratings=20).fit(train, items),
        "Bayesian Average (IMDb)": BayesianAverageRecommender().fit(train, items),
        "Content-based (genres)": ContentBasedRecommender(use_tfidf=True).fit(train, items),
        "Item–Item CF": ItemItemCollaborativeFiltering(k=10).fit(train),
        "Matrix Factorization (SVD)": MatrixFactorizationRecommender(n_factors=20).fit(train),
    }
    profiler = NewUserProfiler(min_support=20).fit(train, items)
    title_map = items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()
    genre_map = items.set_index(config.ITEM_COL)[config.GENRES_COL].to_dict()
    users = sorted(train[config.USER_COL].unique().tolist())
    return train, items, models, profiler, title_map, genre_map, users


train, items, MODELS, PROFILER, TITLE, GENRE, USERS = load_everything()


def fmt(item_id):
    return TITLE.get(item_id, f"#{item_id}")


# --------------------------------------------------------------------------- #
# Sidebar + header
# --------------------------------------------------------------------------- #
st.sidebar.title("🎬 Recommender")
st.sidebar.caption("MovieLens 32M · prototype")
st.sidebar.markdown("---")
st.sidebar.caption("Built for the Recommender Systems individual project.")

st.title("Movie Recommender — interactive demo")

tab_new, tab_user = st.tabs(["✨ Build your profile", "👤 Explore by user"])


# =========================================================================== #
# TAB 1 — questionnaire / cold-start onboarding
# =========================================================================== #
with tab_new:
    st.markdown(
        "New here? Answer a few questions and we'll build your taste profile to "
        "recommend films you'll likely enjoy — no ratings needed (a **cold-start** "
        "onboarding)."
    )
    c1, c2 = st.columns(2)
    with c1:
        q_genres = st.multiselect(
            "1 · Which genres do you enjoy?", GENRE_CHOICES,
            default=["Drama", "Adventure"])
        q_mood = st.selectbox(
            "2 · What are you in the mood for?",
            ["(no preference)"] + list(MOOD_GENRES))
        q_era = st.selectbox("3 · Preferred era?", ERA_CHOICES)
    with c2:
        q_pop = st.radio("4 · Popular hits or hidden gems?", POP_CHOICES,
                         index=1)
        q_quality = st.radio("5 · Quality filter?", QUALITY_CHOICES, index=0)
        q_n = st.slider("How many films?", 5, 20, 10)

    mood = None if q_mood == "(no preference)" else q_mood
    profile_genres = sorted(set(q_genres) | set(MOOD_GENRES.get(mood, [])))
    st.caption("Your taste profile: "
               + (", ".join(profile_genres) if profile_genres else "everything (no preference yet)"))

    recs = PROFILER.recommend(genres=q_genres, mood=mood, era=q_era,
                              popularity=q_pop, quality=q_quality, n=q_n)
    st.markdown(f"### 🍿 Top {len(recs)} films for you")
    if not recs:
        st.warning("No films match all your filters — try relaxing the era or quality filter.")
    else:
        table = pd.DataFrame([{
            "#": i + 1,
            "Film": r["title"],
            "Genres": r["genres"].replace("|", " · "),
            "Avg rating": r["avg_rating"],
            "Match": f"{r['match']:.0%}",
        } for i, r in enumerate(recs)])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("“Match” = cosine similarity between your genre profile and the "
                   "film's genres. Popularity preference and era/quality filters are "
                   "applied on top.")


# =========================================================================== #
# TAB 2 — explore an existing user
# =========================================================================== #
with tab_user:
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    with cc1:
        user_id = st.selectbox("User", USERS, index=0)
    with cc2:
        n = st.slider("How many recommendations", 5, 20, 10, key="user_n")
    with cc3:
        chosen = st.multiselect(
            "Algorithms to compare", list(MODELS),
            default=["Most Popular", "Content-based (genres)", "Item–Item CF",
                     "Matrix Factorization (SVD)"])

    user_rows = train[train[config.USER_COL] == user_id]
    with st.expander(f"👤 Taste profile for user {user_id} "
                     f"({len(user_rows)} ratings, avg {user_rows[config.RATING_COL].mean():.2f})",
                     expanded=True):
        p1, p2 = st.columns([1, 1])
        top_liked = user_rows.sort_values(config.RATING_COL, ascending=False).head(8)
        with p1:
            st.markdown("**Top-rated movies**")
            st.dataframe(
                pd.DataFrame({
                    "Movie": [fmt(i) for i in top_liked[config.ITEM_COL]],
                    "Rating": top_liked[config.RATING_COL].values,
                }),
                hide_index=True, width="stretch")
        with p2:
            st.markdown("**Favourite genres**")
            g = (user_rows.merge(items, on=config.ITEM_COL)[config.GENRES_COL]
                 .str.split("|").explode())
            g = g[g != "(no genres listed)"].value_counts().head(8)
            st.bar_chart(g)

    st.markdown("### Recommendations")
    cols = st.columns(len(chosen)) if chosen else [st]
    for col, name in zip(cols, chosen):
        model = MODELS[name]
        user_recs = model.recommend(user_id, train, n=n, exclude_seen=True)
        with col:
            st.markdown(f"#### {name}")
            if not user_recs:
                st.info("No recommendations.")
                continue
            for rank, item in enumerate(user_recs, 1):
                st.markdown(f"**{rank}. {fmt(item)}**")
                genres = GENRE.get(item, "")
                if genres:
                    st.caption(genres.replace("|", " · "))

    st.markdown("---")
    st.markdown("### 🔎 Find movies similar to…")
    cb = MODELS["Content-based (genres)"]
    options = list(cb.item_ids_)
    labels = {i: fmt(i) for i in options}
    default_idx = int(np.argmax([1 if "Toy Story" in fmt(i) else 0 for i in options])) if options else 0
    pick = st.selectbox("Pick a movie", options, format_func=lambda i: labels[i],
                        index=default_idx)
    if pick is not None:
        sims = cb.similar_items(pick, n=8)
        st.write("Most genre-similar movies:")
        st.dataframe(
            pd.DataFrame({
                "Movie": [fmt(i) for i in sims],
                "Genres": [GENRE.get(i, "").replace("|", " · ") for i in sims],
            }),
            hide_index=True, width="stretch")
