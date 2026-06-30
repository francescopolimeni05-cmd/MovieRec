# Deploying the demo online (Streamlit Community Cloud)

The Streamlit app (`app.py`) can be hosted **for free** on Streamlit Community
Cloud, which runs your app straight from a public GitHub repository.

The app does **not** need the 877 MB MovieLens file online: it runs on the small
bundled data already in this folder —

- `data/movies.csv` (≈ 4 MB) — movie titles & genres
- `data/processed/model_ratings_u10000_i3000_mu20_mi100.pkl` (≈ 27 MB) — the
  cached, filtered 10,000-user working set

`.gitignore` keeps the raw dataset (`ml-32m/`), the old `data/raw/` CSVs and any
stale cache out of the repo, so the push stays light (~31 MB).

---

## 1. Put this folder on GitHub

Create an empty repository on GitHub (e.g. `movie-recommender-demo`), **public**,
with no README. Then, from a terminal **inside this folder**
(`MovieRecommender_Assignment`):

```bash
cd "MovieRecommender_Assignment"
git init
git add .
git commit -m "Movie recommender prototype + Streamlit demo"
git branch -M main
git remote add origin https://github.com/<your-username>/movie-recommender-demo.git
git push -u origin main
```

Sanity check after pushing: on GitHub you should see `app.py`, `src/`,
`requirements.txt`, `data/movies.csv` and `data/processed/...pkl`, **but not**
`ml-32m/` or `data/raw/`.

> If `git push` rejects a large file, confirm `ml-32m/` is ignored — it must not
> be inside this folder, or `.gitignore` must exclude it (it does).

## 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in **with GitHub** (authorize it).
2. Click **Create app** (top-right) → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository**: `<your-username>/movie-recommender-demo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - *(Optional)* **Advanced settings → Python version**: 3.12
4. Click **Deploy**. The first build installs `requirements.txt` and starts the
   app — usually a couple of minutes.

Your app gets a public URL like
`https://<your-username>-movie-recommender-demo.streamlit.app`.

## 3. Updating the live app

Any push to `main` redeploys automatically:

```bash
git add -A && git commit -m "update" && git push
```

---

## Notes & limits

- **Resources**: Community Cloud gives ~1 GB RAM / shared CPU. This app trains its
  six models on the 680k-row cached subset in a few seconds and stays well under
  the limit. Models are built once and held in memory via `@st.cache_resource`, so
  only the **first** page load is slow (~15–30 s).
- **Dependencies**: defined in `requirements.txt` (pandas, numpy, scipy,
  scikit-learn, streamlit). Edit and push to change them; the cloud reinstalls
  automatically.
- **Private repos** also work, but a public repo is simplest for a class demo.
- **Custom data**: to refresh the cached working set, regenerate it locally
  (`python -c "from src.data_loading import get_model_ratings; get_model_ratings(use_cache=False)"`
  with `ml-32m/` present) and commit the new `data/processed/*.pkl`.

Sources: [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) ·
[App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
