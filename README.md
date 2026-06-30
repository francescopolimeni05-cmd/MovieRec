# Individual Recommender Systems Assignment - Placeholder Template

This template provides the project structure and placeholder Python files for the individual assignment.

Students must implement the missing code marked with `TODO`.

## Project goal

Build a movie or music recommender-system prototype including:

1. Data loading and preprocessing
2. Exploratory data analysis
3. Non-personalized baselines
4. Content-based recommendation
5. Collaborative filtering
6. Matrix factorization
7. Evaluation and comparison
8. Final analysis and report

## Recommended datasets

### Movie option
Recommended: MovieLens Latest Small or MovieLens 100K.

Place the files in:

```text
data/raw/
```

Expected MovieLens files:

```text
ratings.csv
movies.csv
```

Expected columns:

```text
ratings.csv: userId, movieId, rating, timestamp
movies.csv: movieId, title, genres
```

### Music option
Students may adapt the template to a music dataset such as Last.fm or another user-song/user-artist interaction dataset.

## Installation

```bash
pip install -r requirements.txt
```

Some optional libraries, such as `scikit-surprise` or `implicit`, may require additional installation depending on the environment.

## How to run

```bash
python main.py
```

The current files are placeholders. The first run will show TODO messages until students complete the code.

## Expected final submission

Students should submit:

- Completed source code
- Notebook or report with explanations
- Evaluation table comparing all methods
- Recommendation examples for at least three users
- Discussion of limitations and possible improvements

## Important

Do not simply fill code mechanically. Explain your design choices and interpret your results.
