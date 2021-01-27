import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import csv

tfidf_profiles = []
profiles_path_csv = 'Part1_Profile_Group2.csv'
rec_path_csv = 'Part1_Recommendation_Group2'


def get_merged():
    df_books = pd.read_csv('bookData_R6C100.csv')
    df_user_ratings = pd.read_csv('UserHistoricalRating_updated.csv')

    df_merged = df_user_ratings.merge(df_books, on='isbn', how='left')
    df_merged['userid'] = df_merged['userid'].astype('string')
    return df_merged


def get_user_profiles():
    df_books = pd.read_csv('bookData_R6C100.csv')
    df_user_ratings = pd.read_csv('UserHistoricalRating_updated.csv')

    df_merged = df_user_ratings.merge(df_books, on='isbn', how='left')
    df_merged['userid'] = df_merged['userid'].astype('string')

    user_books = {}
    for index, row in df_merged.iterrows():
        try:
            user_books[row['userid']].append(row['booktitle'])
        except:
            user_books[row['userid']] = [row['booktitle']]

    return user_books


from sklearn.feature_extraction.text import TfidfVectorizer

for key, value in get_user_profiles().items():
    # if key != '11676':
    #     continue
    user_profile = {'isbn': f'userid: {key}', 'booktitle': "".join(value)}

    books = get_merged()
    books = books[books.userid == key]

    data = pd.read_csv('bookData_R6C100.csv')
    data.head()

    # Extract relevant columns that would influence a book's rating based on book title.
    books_title = data[['isbn', 'booktitle']]

    books_title = books_title.append(user_profile, ignore_index=True)
    books_title.index.name = 'index'

    # REACTIVATE FOR REMOVING READ TITLES
    books_title = books_title[~books_title.index.isin(books.index)]
    books_title.reset_index(inplace=True)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(books_title['booktitle'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # SHOW THE Matrix
    df_matrix = pd.DataFrame(tfidf_matrix.todense(), columns=tf.get_feature_names(), index=books_title['isbn'])
    tfidf_profiles.append(df_matrix.tail(1))

    titles = books_title['booktitle']
    indices = pd.Series(books_title.index, index=books_title['booktitle'])  # converting all titles into a Series


    def book_recommendations(usrprof, n):
        idx = indices[usrprof]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores.pop(len(sim_scores) - 1)
        sim_df = pd.DataFrame.from_records(sim_scores)
        sim_df['isbn'] = books_title['isbn']
        sim_df.columns = ['bookid', 'similarity', 'isbn']
        sim_df.index.name = 'index'
        sim_df.drop(columns=['bookid'], inplace=True, axis=0)
        sim_df.to_csv(f'Part1_SimMatrix_Group2user{key}.csv')
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]

        book_indices = [i[0] for i in sim_scores]
        # return titles.iloc[book_indices]
        rec = books_title
        rec['userid'] = key

        rec = rec.merge(sim_df, on='isbn', how='right')
        rec.drop(columns=['index'], inplace=True)

        return rec.iloc[book_indices]


    # Recommend n books for a book having index 1
    book_index = len(books_title) - 1
    n = 5

    result_df = book_recommendations(books_title.booktitle[book_index], n)
    result_df.index.name = 'index'
    result_df.to_csv(rec_path_csv + f'User{key}.csv')
    print(result_df)

df_profiles = pd.concat(tfidf_profiles)
df_profiles.to_csv(profiles_path_csv)
