# Kodhjälp från ChatGPT: prompt: "Refactor the code to make it more readable and modular. Use functions to encapsulate logic and improve maintainability. Add docstrings for each function to explain its purpose and parameters. Ensure that the code adheres to PEP 8 style guidelines for Python code."
"""
Refactored Movie Recommender Module

Provides a MovieRecommender class for use in scripts or notebooks.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def rensa_text(text: str) -> str:
    """Standardize text: remove punctuation, extra spaces, title case."""
    cleaned = re.sub(r"[^\w\s]", " ", str(text)).title()
    return " ".join(cleaned.split())


def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="movieId")
    df["title"] = df["title"].map(rensa_text)
    df["genres"] = df["genres"].map(lambda g: g.replace('|', ' ')).map(rensa_text)
    # Extract year if in parentheses in title
    year = df["title"].str.extract(r"\((\d{4})\)")
    df["year"] = pd.to_numeric(year[0], errors="coerce").fillna(0).astype(int)
    return df


def load_tags(path: str, min_count: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["tag"])
    df["tag"] = df["tag"].map(rensa_text).str.lower()
    counts = df["tag"].value_counts()
    keep = counts[counts >= min_count].index
    return df[df["tag"].isin(keep)]


def load_ratings(path: str, min_movie_ratings: int = 20) -> pd.DataFrame:
    df = pd.read_csv(path)
    movie_counts = df["movieId"].value_counts()
    valid = movie_counts[movie_counts >= min_movie_ratings].index
    return df[df["movieId"].isin(valid)]


def build_content_matrix(movies: pd.DataFrame, tags: pd.DataFrame, max_features: int = 5000):
    tag_str = tags.groupby("movieId")["tag"].apply(lambda toks: " ".join(toks))
    tag_str = tag_str.reindex(movies["movieId"]).fillna("")
    corpus = (movies["genres"].fillna("") + " " + tag_str).tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vectorizer.fit_transform(corpus)
    return tfidf, vectorizer


def build_cf_model(ratings: pd.DataFrame, n_neighbors: int = 10):
    pivot = ratings.pivot_table(index='movieId', columns='userId', values='rating', fill_value=0)
    sparse = csr_matrix(pivot.values)
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model.fit(sparse)
    return model, pivot.index.tolist(), sparse


class MovieRecommender:
    """Encapsulates loading data, building models, and generating recommendations."""
    def __init__(
        self,
        movies_path: str,
        tags_path: str,
        ratings_path: str,
        tag_min_count: int = 3,
        min_movie_ratings: int = 20,
        tfidf_features: int = 5000,
        knn_neighbors: int = 10
    ):
        self.movies = load_movies(movies_path)
        self.tags = load_tags(tags_path, min_count=tag_min_count)
        self.ratings = load_ratings(ratings_path, min_movie_ratings)

        self.tfidf_matrix, self.vectorizer = build_content_matrix(
            self.movies, self.tags, max_features=tfidf_features
        )
        self.cb_model = NearestNeighbors(
            metric='cosine', algorithm='brute', n_neighbors=knn_neighbors
        )
        self.cb_model.fit(self.tfidf_matrix)

        self.cf_model, self.movie_ids, self.sparse_matrix = build_cf_model(
            self.ratings, n_neighbors=knn_neighbors
        )

    def recommend_cb(self, title: str, k: int = 5) -> list:
        idx_list = self.movies.index[self.movies['title'] == title].tolist()
        if not idx_list:
            return []
        idx = idx_list[0]
        distances, indices = self.cb_model.kneighbors(
            self.tfidf_matrix[idx], n_neighbors=k+1
        )
        recs = [self.movies.iloc[i]['title'] for i in indices.flatten() if i != idx]
        return recs[:k]

    def recommend_cf(self, title: str, k: int = 5) -> list:
        movie_id = self.movies.loc[self.movies['title'] == title, 'movieId'].squeeze()
        if movie_id not in self.movie_ids:
            return []
        idx = self.movie_ids.index(movie_id)
        distances, indices = self.cf_model.kneighbors(
            self.sparse_matrix[idx], n_neighbors=k+1
        )
        rec_ids = [self.movie_ids[i] for i in indices.flatten() if i != idx]
        return self.movies[self.movies['movieId'].isin(rec_ids)]['title'].tolist()

    def recommend_hybrid(self, title: str, k: int = 5, weights: tuple = (0.7, 0.3)) -> list:
        cb = self.recommend_cb(title, k*3)
        cf = self.recommend_cf(title, k*3)
        scores = {}
        for rank, t in enumerate(cb):
            scores[t] = scores.get(t, 0) + weights[0] * (k*3 - rank)
        for rank, t in enumerate(cf):
            scores[t] = scores.get(t, 0) + weights[1] * (k*3 - rank)
        sorted_titles = sorted(scores.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_titles[:k]]

    def recommend(
        self,
        title: str,
        method: str = 'cb',
        k: int = 5,
        **kwargs
    ) -> list:
        """General entry point: method in {'cb','cf','hybrid'}."""
        if method == 'cb':
            return self.recommend_cb(title, k)
        elif method == 'cf':
            return self.recommend_cf(title, k)
        elif method == 'hybrid':
            return self.recommend_hybrid(title, k, kwargs.get('weights', (0.7, 0.3)))
        else:
            raise ValueError(f"Unknown method: {method}")
