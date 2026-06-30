# Movie Recommender System — Project Report

**Individual Project · Recommender Systems · Esade**
Track: **Movies** · Dataset: **MovieLens 32M**

---

## 1. Introduction

The goal of this project is to build a single movie-recommender prototype and grow
it from simple non-personalized baselines to learned latent-factor models, keeping
one dataset, one train/test protocol and one set of metrics so that every method is
directly comparable. Beyond raw accuracy, the project also measures **catalog
coverage** and **novelty** to expose the popularity bias that dominates offline
evaluation.

All algorithms live in the `src/` package and share the same interface
(`fit(...)` → `recommend(user, n)`), so they are interchangeable in the pipeline
(`main.py`) and in the walkthrough notebook.

## 2. Dataset description

MovieLens 32M: **200,948 users**, **84,432 rated movies** (87,585-movie catalog),
**32,000,204 ratings** on a half-star scale (0.5–5.0), plus genre metadata and a
`tags.csv` file. The user–item matrix is **99.81% sparse**. Ratings skew positive
(mean ≈ 3.54, ~159 ratings/user, ~379/item) and item popularity follows a strong
long tail — a handful of blockbusters absorb most ratings while most films are
rarely rated.

## 3. Preprocessing and EDA

EDA (rating distribution, long-tail popularity, genre frequency, most active users /
most popular items) is computed on the **full** 32M dataset (loaded with compact
dtypes, ~0.6 GB RAM). For modelling we then apply, in order:

- **Catalog cap**: keep the `config.MAX_ITEMS = 3000` most-popular movies, which
  bounds the size of the item–item similarity matrix and the MF model.
- **k-core filtering**: keep items with ≥ 100 ratings and users with ≥ 20 ratings.
- **User sub-sampling** (`config.MAX_USERS = 10000`), followed by re-pruning items to
  ≥ 10 ratings so the matrices stay dense enough to learn from. The working set is
  **~1.34M ratings** (10,000 users × 3,000 items) with a median of ~100 ratings/item,
  cached to `data/processed/` so re-runs skip the 877 MB load. This is a deliberate,
  reversible sampling choice — set `MAX_USERS = MAX_ITEMS = None` to run on the full
  filtered data (see *Scalability* below).
- **Per-user temporal 80/20 split** (default): each user's most recent 20% of
  ratings (by timestamp) are held out for test. This avoids the
  train-on-future / test-on-past leakage of a random split and is the realistic
  protocol; a random split is kept as an ablation. Both keep every test user in
  train, as the personalized models require.

## 4. Algorithms implemented

| Family | Model | Idea |
|---|---|---|
| Non-personalized | Most Popular | Rank by interaction count |
| Non-personalized | Highest Average (min 20) | Best mean rating with a hard minimum-support cut-off |
| Non-personalized | Bayesian Average (IMDb) | Damped weighted rating `WR = v/(v+m)·R_j + m/(v+m)·C` (course formula) |
| Non-personalized | Random | Lower-bound / coverage ceiling |
| Content-based | TF-IDF **and** raw-genre vectors | User profile = mean-centered, rating-weighted sum of item vectors; cosine scoring (both weightings compared) |
| Collaborative | Item–Item CF (k=10) | Adjusted-cosine item similarity, top-k neighbourhood |
| Collaborative | User–User CF (k=40) | Mean-centered user similarity, top-k neighbours |
| Latent factor | Matrix Factorization | **Truncated SVD** (`sklearn.decomposition.TruncatedSVD`): `R ≈ U·Vᵀ`, prediction = `pᵤ·qᵢ`, top-N by `argsort` |

The matrix-factorization model follows the course notebooks exactly: the sparse
user–item matrix is factorized with **`sklearn.decomposition.TruncatedSVD`**, a rating
is predicted as the dot product of the user and item latent vectors, and the top-N
unseen items are taken with `argsort`. The neighbourhood size *k* and the number of
latent factors were chosen with a small **tuning sweep** (`tuning.py`,
`results/tuning.csv`): item–item CF is best with a small *k* (k=10), and Truncated SVD
peaks around 20 latent factors before over-fitting — so the pipeline uses k=10 and 20
factors.

## 5. Evaluation protocol

For each user, a held-out item is **relevant** if its test rating is ≥ 4.0. We
generate top-10 recommendations (excluding items already seen in training) and
average per-user metrics over a fixed user sample, identical for every model. We
report the **temporal-split** numbers (leakage-free):

- **Accuracy (ranking)**: Precision@10, Recall@10, Hit-Rate@10, NDCG@10, MRR, with
  a **95% confidence interval** on NDCG (standard error over users).
- **Beyond-accuracy**: catalog coverage, novelty (mean self-information), and a
  **long-tail / popularity-bias** analysis (`analysis_longtail.py`).
- **Rating prediction**: MAE, RMSE (item–item CF, which predicts on the rating scale).

## 6. Results

Top-10, temporal split, 95% CI on NDCG (10,000-user working set, ~344 evaluated):

| Model | Prec@10 | Rec@10 | NDCG@10 (±95% CI) | MRR | Coverage | Novelty |
|---|---|---|---|---|---|---|
| **Matrix Factorization (SVD)** | **0.101** | **0.108** | **0.138 ± 0.022** | **0.240** | 0.135 | 2.35 |
| Most Popular | 0.058 | 0.059 | 0.075 ± 0.014 | 0.151 | 0.027 | 1.51 |
| Bayesian Average (IMDb) | 0.038 | 0.045 | 0.058 ± 0.013 | 0.125 | 0.015 | 2.38 |
| Item–Item CF (k=10) | 0.030 | 0.037 | 0.037 ± 0.008 | 0.082 | 0.410 | 4.36 |
| Highest Average (min 20) | 0.014 | 0.016 | 0.016 ± 0.005 | 0.031 | 0.006 | 5.72 |
| User–User CF (k=40) | 0.011 | 0.015 | 0.013 ± 0.005 | 0.027 | **0.502** | 4.96 |
| Content-based (TF-IDF) | 0.002 | 0.003 | 0.003 ± 0.003 | 0.006 | 0.091 | 12.51 |
| Content-based (raw genres) | 0.001 | 0.001 | 0.001 ± 0.001 | 0.002 | 0.074 | 12.66 |
| Random | 0.000 | 0.000 | 0.000 | 0.000 | 0.032 | **13.07** |

Rating prediction (item–item CF): **MAE = 0.68, RMSE = 0.91** on the 0.5–5 scale
(`results/rating_prediction.csv`). *(Ranking metrics + CI in `results/metrics.csv`;
chart in `results/figures/metrics_comparison.png`.)*

The headline result has two layers. First, **matrix factorization (Truncated SVD) is
the clear overall winner** — its NDCG (0.138 ± 0.022) is significantly above Most
Popular (0.075 ± 0.014; the confidence intervals do not overlap), confirming why the
course treats latent-factor models as the workhorse of recommendation. (These
temporal-split numbers are markedly lower than the random-split ones we first
obtained — see *Threats to validity* — because removing the leakage makes the task
realistically harder.) Second, the classic **accuracy vs. discovery trade-off** holds
among the rest: Most Popular and the Bayesian average are accurate but narrow
(~2–3% coverage), while collaborative methods cover 41–50% of the catalog and
content-based surfaces the most novel items.
Two course-aligned details stand out: the **Bayesian (IMDb) average clearly beats the
hard-threshold "highest average"** (NDCG 0.058 vs 0.016), and **TF-IDF vs. raw genre
vectors** make almost no difference for content-based here (genres are already short,
near-binary documents). The ranking is stable across sample sizes (3,000 / 5,000 /
10,000 users) and across evaluation samples (the 95% CIs are tight).

## 7. Recommendation examples

For the same user (id 124204), each method has a recognizable "personality" (full
table in `results/example_recommendations.csv`):

- **Most Popular** → safe mainstream hits (Jurassic Park, Raiders of the Lost Ark,
  …).
- **Bayesian Average (IMDb)** → critically-acclaimed classics rather than the most-
  rated ones (The Usual Suspects, Dr. Strangelove, …).
- **Content-based** → genre-coherent but obscure items (high novelty, low accuracy).
- **Item–Item CF** → eclectic taste neighbours (Planet Earth II, Lawrence of Arabia,
  Capturing the Friedmans, The King's Speech, …).
- **Matrix Factorization (SVD)** → strongly personalized quality picks the user had
  not seen (Memento, The Usual Suspects, Raiders of the Lost Ark, …) — the best blend
  of relevance and personalization, though still popular-leaning (see below).

## 8. Threats to validity

We are deliberately explicit about what these numbers do and do not show.

- **Evaluation protocol (addressed).** A random per-user split leaks information
  (training on a user's future, testing on their past). We therefore default to a
  **temporal** split (hold out each user's most recent ratings). The effect is large:
  SVD's NDCG drops from ~0.33 (random) to ~0.12 (temporal). The temporal numbers above
  are the ones we trust.
- **Statistical reliability (addressed).** Metrics are point estimates over a user
  sample, so we report a **95% confidence interval** on NDCG (standard error over
  users). The top models' intervals do not overlap, so the ranking is significant, not
  noise.
- **Popularity bias / cold-start (measured).** Offline accuracy is structurally biased
  toward popular items. `analysis_longtail.py` quantifies this: of each model's top-10,
  the share that falls in the catalogue's long tail (bottom 80% by popularity) is
  **0% for Most Popular, ~1% for SVD, ~2% for Bayesian, ~49% for item–item CF and ~97%
  for content-based**. So the SVD "winner" earns much of its accuracy by staying on the
  head — it is *not* a strong item-cold-start model. True cold users/items are not
  *evaluated offline* (the split guarantees overlap); in the product, the **new-user
  cold-start is handled by the questionnaire onboarding in the Streamlit app**, which
  builds a content profile from a few answers (no ratings needed).
- **Optimistic absolute numbers.** Modelling runs on a **dense, popular subset**
  (top-3,000 items, active users); this makes accuracy look higher than a full-catalogue
  system would and means the figures are **not comparable to published MovieLens
  benchmarks**. The *relative* ranking is the trustworthy part.
- **SVD imputation-by-zero.** Truncated SVD on the raw matrix treats missing entries as
  0 ("dislike"), the course's pedagogical choice; an MF trained only on observed
  ratings (mean-centered SVD or biased SGD) is more principled for explicit feedback.
- **Other.** Content features are **genres only**; the **relevance threshold (≥ 4)** is
  a modelling choice; only **explicit** ratings are used.

**Scalability note.** Neighbourhood CF stores an **O(n²)** dense similarity matrix
(≈160 billion entries for all 200,948 users), so item/user CF set the tractability
ceiling and motivate the bounded working set (10,000 users × 3,000 items ≈ 1.34M
ratings). Truncated SVD and the popularity baselines would scale much further. The
dense CF caches use **float32** and derive the per-user "rated" mask on the fly, which
keeps the deployed app's peak memory at **≈0.7 GB** — within the free Streamlit tier.

## 9. Conclusion

Under a leakage-free temporal split, **matrix factorization (Truncated SVD) is the
best single model** by a statistically significant margin — which is why latent-factor
methods are the backbone of modern recommenders and the focus of the course. But the
long-tail analysis tempers that headline: SVD wins largely by recommending **popular**
items (~1% long-tail share), so it is strong for engagement, weak for discovery. There
is no universal winner: accuracy-oriented models (SVD, popularity, Bayesian average)
are narrow, while CF and content-based trade accuracy for far higher coverage and
novelty. The right choice depends on the objective — engagement, discovery or coverage.
An interactive **Streamlit app** (`app.py`) offers two modes: a **new-user
questionnaire** that solves the cold-start problem with a content profile built from a
few answers, and an **existing-user explorer** that compares all algorithms side by
side. Beyond the tuning, CIs and bias analysis already done, natural next steps are a
hybrid that blends SVD accuracy with content/CF novelty, explicit diversity /
de-biasing, a proper offline cold-start evaluation, and deploying the demo online.

---

### How to reproduce

```bash
pip install -r requirements.txt
python main.py              # full pipeline (temporal split) -> metrics.csv + rating_prediction.csv
python tuning.py            # k / latent-factor sweep -> results/tuning.csv
python analysis_longtail.py # popularity-bias / long-tail share -> results/popularity_bias.csv
python generate_eda.py      # EDA + comparison figures -> results/figures/
streamlit run app.py        # interactive demo UI
pip install jupyter && jupyter notebook notebooks/recommender_walkthrough.ipynb
```

To put the demo **online** (free Streamlit Community Cloud), see `DEPLOY.md`: the app
runs on the small bundled `data/movies.csv` + `data/processed/*.pkl` cache, so the
877 MB dataset is not needed on the server.

The MovieLens 32M dataset is expected in `../ml-32m/` (next to this folder);
the path is configurable in `src/config.py`.
