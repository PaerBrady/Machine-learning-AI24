import pandas as pd
import numpy as np
from nltk.corpus import words
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Konverterar text till titel-format och tar bort skiljetecken och extra mellanslag
rensa_text = lambda text: " ".join(re.sub(r"[^\w\s]", " ", str(text).title()).split())
# Rensa och strukturera kolumnerna:
df_filmer = pd.read_csv("movies.csv").drop_duplicates("title").assign(
    title=lambda df: df["title"].map(rensa_text),
    genres=lambda df: df["genres"].map(rensa_text),
    # Extrahera årtal från titel
    år=lambda df: df["title"].str.extract(r"(\b\d{4}\b)").astype(float).fillna(0)
)

# Hämta en uppsättning engelska ord (för att filtrera bort nonsensord)
eng_ord_set = set(words.words())
df_taggar = pd.read_csv("tags.csv").dropna()

# Rensa varje tagg och behåller bara taggar som finns i listan över engelska ord och har fler än 2 tecken
df_taggar = df_taggar[df_taggar["tag"].map(rensa_text).str.lower().isin(eng_ord_set) & 
                      df_taggar["tag"].str.len().gt(2)]

# Omvandla timestamp till datum (utan tidskomponent)
df_taggar["datum"] = pd.to_datetime(df_taggar["timestamp"], unit="s").dt.date

# Grupper data per användare för att räkna antal unika dagar(aktiva_dagar) och antalt antal taggar(antal_taggar)
gruppering = df_taggar.groupby("userId").agg(
    aktiva_dagar=("datum", "nunique"), antal_taggar=("tag", "count"))

# Slå ihop beräkningarna med originaldatan och beräkna genomsnittligt antal taggar per dag för varje användare
df_taggar = df_taggar.merge(gruppering, on="userId").assign(
    taggar_per_dag=lambda df: df["antal_taggar"] / df["aktiva_dagar"])

# Filtrera bort användare utifrån antal taggar per dag, antal aktiva dagar och antal totala taggar
df_taggar = df_taggar.query("5 < taggar_per_dag < 45 & 20 < aktiva_dagar < 1000 & 100 < antal_taggar < 5000")

# Slå ihop taggar med filmer och grupper taggar per film och sammanfoga dem till en enda sträng per film.
df_filmer = df_filmer.merge(df_taggar.groupby("movieId")["tag"].agg(" ".join).reset_index(), 
                            on="movieId", how="left").fillna("")# Om filmen saknas, ersätta den med en tom sträng

# Omvandla timestamp till datum och beräkna genomsnittligt betyg för varje film, avrundat till 2 decimaler.
df_betyg = pd.read_csv("ratings.csv").assign(
    datum=lambda df: pd.to_datetime(df["timestamp"], unit="s").dt.date,
    medelbetyg=lambda df: df.groupby("movieId")["rating"].transform("mean").round(2)
)
# Grupper betygsdata per användare för att räkna totalt antal betyg och antal unika dagar 
gruppering_betyg = df_betyg.groupby("userId").agg(
    antal_betyg=("rating", "count"), aktiva_dagar=("datum", "nunique"))
# Slå ihop beräkningarna med originaldatan och beräkna genomsnittligt antal betyg per dag.
df_betyg = df_betyg.merge(gruppering_betyg, on="userId").assign(
    betyg_per_dag=lambda df: df["antal_betyg"] / df["aktiva_dagar"])
# Filtrera bort användare utifrån betyg, aktiva daga och betyg per dag
df_betyg = df_betyg.query("50 < antal_betyg < 4000 & 20 < aktiva_dagar < 1800 & 1 <= betyg_per_dag < 9")

# Slå ihop filmer med deras genomsnittliga betyg & normalisera genomsnittliga betyget
df_filmer = df_filmer.merge(df_betyg[["movieId", "medelbetyg"]].drop_duplicates(),
                             on="movieId", how="left").fillna(0.0)
df_filmer["medelbetyg"] = MinMaxScaler().fit_transform(df_filmer[["medelbetyg"]])

# Skapa en textrepresentation för varje film genom att kombinera: Genres, Årtal, Taggar och Medelbetyg
df_filmer["text_representation"] = df_filmer[["genres", "år", "tag", "medelbetyg"]].astype(str).agg(" ".join, axis=1)
# Skapa en TF-IDF-matris baserat på filmens textrepresentation oc ignorera vanliga engelska ord (stopwords)
# Begränsa till de 5000 mest relevanta orden
tfidf_matriser = TfidfVectorizer(
    stop_words="english", max_features=5000).fit_transform(df_filmer["text_representation"])

# Rekommendationsfunktioner
def rekommendera(title, antal=5, typ="text"):
    if title not in df_filmer["title"].values:
        return "Filmen finns inte i databasen."
    # Hitta index för filmen i df_filmer
    title_index = df_filmer.index[df_filmer["title"] == title][0]
    # Beräkna likhet beroende på vald metod:
    if typ == "text":
        # Använd TF-IDF och cosinuslikhet för att hitta filmer med liknande innehåll (genrer, taggar, medelbetyg, årtal)
        likhet = cosine_similarity(tfidf_matriser[title_index], tfidf_matriser).flatten()
    else:
        # Använd betygsdata och cosinuslikhet för att hitta filmer med liknande betygsmönster
        betygsmatris = df_betyg.pivot(index="userId", columns="movieId", values="rating").fillna(0)

        # Hitta filmens movieId
        movie_id = df_filmer.loc[df_filmer["title"] == title, "movieId"].values[0]

        # Beräkna cosinuslikhet mellan filmen och alla andra filmer i betygsmatrisen
        likhet = cosine_similarity(
            betygsmatris.loc[:, movie_id].values.reshape(1, -1),  # Filmens betygsmönster
            betygsmatris.T )[0] # Alla andra filmernas betygsmönster
        
    # Sortera filmer efter högsta likhet och returnera de bästa matcherna
    return df_filmer.iloc[np.argsort(-likhet)][1:antal+1]["title"].tolist()
        
# Kombinerar innehållsbaserad filtrering (text) och kollaborativ filtrering (betyg).
def hybrid_rekommendation(title, antal=5):
    # Hämta rekommendationer från både metoderna och ta fram dubbelt så många för bättre urval
    genre_based, rating_based = rekommendera(title, antal*2, "text"), rekommendera(title, antal*2, "betyg")
    # Skapa en poäng för varje film baserat på hur ofta den förekommer i rekommendationerna
    movie_scores = {
        m: (2 if m in genre_based else 1) + rating_based.count(m) for m in genre_based + rating_based}
    # Sortera filmer efter högst poäng sedan om två filmer har samma poäng, prioritera filmer från genre_based-listan
    sorted_movies = sorted(movie_scores, 
                           key=lambda x: (-movie_scores[x], 
                                          genre_based.index(x) if x in genre_based else float('inf')))
    return f"Five movies that match '{title}':\n" + "\n".join([f"{i+1}: {m}" for i, m in enumerate(sorted_movies[:antal])])