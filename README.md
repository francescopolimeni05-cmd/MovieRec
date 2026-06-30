# Movie Recommender System

A movie-recommender prototype built for the *Recommender Systems* individual
project (Esade). It implements the full classical toolbox on a single MovieLens
dataset — from non-personalized baselines to a latent-factor model — and compares
the methods under one consistent, leakage-free evaluation protocol.

**Live demo:** https://movierec-fp.streamlit.app/

## What it does

The project trains and compares nine recommenders on MovieLens 32M: a Most-Popular
and a damped Bayesian (IMDb) baseline, a content-based model over movie genres,
item–item and user–user collaborative filtering, and a Truncated-SVD matrix
factorization. Every model shares the same interface and is evaluated on the same
temporal train/test split, so the comparison is apples-to-apples.

Evaluation goes beyond raw accuracy: alongside Precision/Recall/NDCG/MRR (with
95% confidence intervals) it reports catalog coverage, novelty, a long-tail
popularity-bias analysis, and rating-prediction error (MAE/RMSE).

The interactive Streamlit app has two modes: an onboarding **questionnaire** that
builds a taste profile for a brand-new user (a cold-start solution), and an
**explorer** that compares every algorithm's recommendations for an existing user.

## Key result

Under a leakage-free temporal split, the Truncated-SVD model is the strongest
single method (NDCG@10 = 0.138 ± 0.022), significantly ahead of the popularity
baseline. The long-tail analysis qualifies that headline, though: SVD earns its
accuracy mostly on popular titles, while collaborative and content-based methods
cover far more of the catalog — the classic accuracy-vs-discovery trade-off.

## Running it locally

```bash
pip install -r requirements.txt

streamlit run app.py        # interactive demo
python main.py              # full training + evaluation pipeline
python tuning.py            # hyper-parameter sweep (k, latent factors)
python analysis_longtail.py # popularity-bias analysis
python generate_eda.py      # EDA and comparison figures
```

The app runs out of the box on the small bundled data in `data/`. The full
pipeline (`main.py`, `generate_eda.py`) additionally expects the MovieLens 32M
dataset in `../ml-32m/`; the path is configurable in `src/config.py`.

## Repository layout

```
app.py                  Streamlit application
main.py                 end-to-end training + evaluation pipeline
tuning.py               k / latent-factor sweep
analysis_longtail.py    popularity-bias (long-tail) analysis
generate_eda.py         EDA and comparison figures
src/                    models, data loading, evaluation metrics
notebooks/              annotated walkthrough notebook
results/                metrics, figures and example recommendations
data/                   bundled movie metadata + cached working set
REPORT.md               written report
```

## Notes on method and data

To keep neighbourhood CF and the live app tractable, the models are trained on a
bounded working set (the 10,000 most-active users over the 3,000 most-popular
items, ≈1.34M ratings); exploratory analysis is still computed on the full 32M
dataset. The matrix-factorization model follows the Truncated-SVD approach used in
the course. Evaluation choices and their limitations are discussed in `REPORT.md`.

## Author

Francesco Polimeni — Recommender Systems, Esade.
