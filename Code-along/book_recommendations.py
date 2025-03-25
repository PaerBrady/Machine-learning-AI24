import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_data_files():
    books = pd.read_csv("../data/Books.csv")
    ratings = pd.read_csv("../data/Ratings.csv")
    books.drop_duplicates("Book-Title",inplace=True)
    merged = ratings.merge(books, on="ISBN")
    merged.drop(["ISBN", "Image-URL-S", "Image-URL-M", "Image-URL-L", "Publisher"], axis=1, inplace=True)
    merged.dropna(inplace=True)
    return books, merged

def extract_features(raw_table):
    x = raw_table.groupby("User-ID").count()["Book-Rating"] > 100
    expert_users = x[x].index

    filtered_ratings = raw_table[raw_table["User-ID"].isin(expert_users)]

    y = filtered_ratings.groupby("Book-Title").count()["Book-Rating"] >= 50
    famous_books = y[y].index
    user_ratings = filtered_ratings[filtered_ratings["Book-Title"].isin(famous_books)]

    design_matrix = user_ratings.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
    design_matrix.fillna(0, inplace=True)
    return design_matrix

def make_model(matrix):
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    sim_score = cosine_similarity(scaled)
    return sim_score

def recommend(book_name, table, design_matrix, similarity_score):
    index = np.where(design_matrix.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

    data = []
    for index, _ in similar_books:
        item = []
        temp_df = table[table["Book-Title"]==design_matrix.index[index]]
        item.extend(temp_df["Book-Title"].values)
        item.extend(temp_df["Book-Author"].values)
        data.append(item)
    return data

def main():
    books, table = load_data_files()
    matrix = extract_features(table)
    model = make_model(matrix)
    name = input("Input a book name: ")
    print(recommend(name, books, matrix, model))

if __name__ == '__main__':
    main()