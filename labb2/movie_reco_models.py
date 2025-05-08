#!/usr/bin/env python3
"""
Refactored Movie Recommender Module

Provides a MovieRecommender class for use in scripts or notebooks.
"""
import os 
import warnings
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, coo_matrix

# Suppress pandas PerformanceWarnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def rensa_text(text: str) -> str:
    """
    Standardize a text string by:
      1. Removing any non-alphanumeric characters (except whitespace).
      2. Converting to title case.
      3. Collapsing multiple spaces into one.

    Parameters:
      text (str): Input string to clean.

    Returns:
      str: Cleaned text.
    """
    cleaned = re.sub(r"[^\w\s]", " ", str(text)).title()
    return " ".join(cleaned.split())


def load_movies(path: str) -> pd.DataFrame:
    """
    Load and preprocess the movies dataset, extracting year and cleaning title.

    Steps:
      - Expand user path (~).
      - Read CSV and drop duplicate movieId.
      - Extract four-digit year from original title.
      - Remove year substring from title.
      - Clean title and genres.

    Parameters:
      path (str): Path to movies.csv.

    Returns:
      pd.DataFrame: Movies with columns ['movieId','title','genres','year',...].
    """
    path = os.path.expanduser(path)
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="movieId")

    # Extract year before cleaning title
    years = df["title"].str.extract(r"\((\d{4})\)")
    df["year"] = pd.to_numeric(years[0], errors="coerce").fillna(0).astype(int)

    # Remove the year portion (e.g. '(1999)') from title string
    df["title"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    # reorder 'X, The' → 'The X' and reoder 'X The' → 'The X'
    df["title"] = df["title"].str.replace(r"^(.+),\s*(The|An|A)$", r"\2 \1", regex=True)
    
    # Clean titles and genres
    df["title"] = df["title"].map(rensa_text)
    df["genres"] = df["genres"].map(lambda g: g.replace('|', ' ')).map(rensa_text)

    return df


def load_tags(path: str, min_count: int = 3) -> pd.DataFrame:
    path = os.path.expanduser(path)
    df = pd.read_csv(path).dropna(subset=["tag"])
    df["tag"] = df["tag"].map(rensa_text).str.lower()
    counts = df["tag"].value_counts()
    keep = counts[counts >= min_count].index
    return df[df["tag"].isin(keep)]


def load_ratings(path: str, min_movie_ratings: int = 20) -> pd.DataFrame:
    path = os.path.expanduser(path)
    df = pd.read_csv(path)
    movie_counts = df["movieId"].value_counts()
    valid = movie_counts[movie_counts >= min_movie_ratings].index
    return df[df["movieId"].isin(valid)]


def build_content_matrix(movies: pd.DataFrame, tags: pd.DataFrame, max_features: int = 5000):
    tag_str = tags.groupby("movieId")["tag"].apply(lambda toks: " ".join(toks))
    tag_str = tag_str.reindex(movies["movieId"]).fillna("")
    genres = movies["genres"].fillna("").astype(str)
    corpus = (genres + " " + tag_str).tolist()
    corpus = [doc if isinstance(doc, str) else "" for doc in corpus]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = vectorizer.fit_transform(corpus)
    return tfidf, vectorizer


def build_cf_model(ratings: pd.DataFrame, n_neighbors: int = 10):
    movie_ids = sorted(ratings['movieId'].unique())
    user_ids = sorted(ratings['userId'].unique())
    m2i = {m: i for i, m in enumerate(movie_ids)}
    u2i = {u: i for i, u in enumerate(user_ids)}
    rows = ratings['movieId'].map(m2i)
    cols = ratings['userId'].map(u2i)
    data = ratings['rating'].values
    sparse = coo_matrix((data, (rows, cols)),
                        shape=(len(movie_ids), len(user_ids))).tocsr()
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model.fit(sparse)
    return model, movie_ids, sparse


class MovieRecommender:
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
        idxs = self.movies.index[self.movies['title'] == title].tolist()
        if not idxs:
            return []
        idx = idxs[0]
        dists, inds = self.cb_model.kneighbors(
            self.tfidf_matrix[idx], n_neighbors=k+1
        )
        recs = [self.movies.iloc[i]['title'] for i in inds.flatten() if i != idx]
        return recs[:k]

    def recommend_cf(self, title: str, k: int = 5) -> list:
        movie_id = self.movies.loc[self.movies['title'] == title, 'movieId'].squeeze()
        if movie_id not in self.movie_ids:
            return []
        idx = self.movie_ids.index(movie_id)
        dists, inds = self.cf_model.kneighbors(
            self.sparse_matrix[idx], n_neighbors=k+1
        )
        rec_ids = [self.movie_ids[i] for i in inds.flatten() if i != idx]
        return self.movies[self.movies['movieId'].isin(rec_ids)]['title'].tolist()

    def recommend_hybrid(self, title: str, k: int = 5, weights: tuple = (0.7, 0.3)) -> list:
        cb = self.recommend_cb(title, k*3)
        cf = self.recommend_cf(title, k*3)
        scores = {}
        for r, t in enumerate(cb): scores[t] = scores.get(t, 0) + weights[0]*(k*3-r)
        for r, t in enumerate(cf): scores[t] = scores.get(t, 0) + weights[1]*(k*3-r)
        return [t for t,_ in sorted(scores.items(), key=lambda x: -x[1])[:k]]

    def recommend(self, title: str, method: str = 'cb', k: int = 5, **kwargs) -> list:
        if method == 'cb': return self.recommend_cb(title, k)
        if method == 'cf': return self.recommend_cf(title, k)
        if method == 'hybrid': return self.recommend_hybrid(title, k, kwargs.get('weights',(0.7,0.3)))
        raise ValueError(f"Unknown method: {method}")
