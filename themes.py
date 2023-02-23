from pollbot import source
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import nltk
nltk.download('punkt')
pd.set_option('display.max_colwidth', 100)


df = pd.read_csv(source, sep='.')  # add data here

stopwords.words('english')


def clean_data(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocesses a given string of text by removing special characters, multiple space characters, and optionally, stopwords.

    Args:
        text (str): A string of text to be preprocessed.
        remove_stopwords (bool, optional): A boolean value indicating whether or not to remove stopwords from the text. Defaults to True.

    Returns:
        str: The preprocessed string of text.

    Example:
        >>> preprocess("This is an example sentence! It includes special characters, such as @ and $.", True)
        'example sentence includes special characters'
    """
    # remove special characters
    text = re.sub("[^A-Za-z]+", " ", text)

    # remove multiple space characters
    text = re.sub("\s+", " ", text)

    # remove stopwords
    if remove_stopwords == True:
        # tokenizing
        tokens = word_tokenize(text)
        # check for stopwords
        tokens = [word for word in tokens if not word.lower()
                  in stopwords.words("english")]
        text = " ".join(tokens)

    text = text.lower()
    return text


def vectorize_tfidf(df):
    """
    Vectorizes and scales the given text data using the TfidfVectorizer.

    Args:
        df (pd.Series): A pandas Series of text data.

    Returns:
        X (np.array): A numpy array of the vectorized text data.
        vec (TfidfVectorizer): The fitted TfidfVectorizer object.
    """
    vec = TfidfVectorizer(sublinear_tf=True, analyzer='word',
                          norm='l2', min_df=5, max_df=0.95)
    X_vec = vec.fit_transform(df)  # fitting and transforming the data

    scaler = StandardScaler(with_mean=False)  # scaling the data
    X_fit = scaler.fit(X_vec)
    X = scaler.transform(X_vec)

    return (X, vec)


df['cleaned'] = df['corpus'].apply(
    lambda x: clean_data(x, remove_stopwords=True))
X, vec = vectorize_tfidf(df['cleaned'])

kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
df['cluster'] = kmeans.labels_


def get_top_words_per_cluster(df, vec, n_words):
    """
    Gets the top n words per cluster.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the text data and cluster labels.
        vec (TfidfVectorizer): A fitted TfidfVectorizer object.
        n_words (int): The number of top words to return per cluster.

    Returns:
        top_words (dict): A dictionary containing the top n words per cluster.
    """
    top_words = {}
    for cluster in range(0, 5):
        top_words[cluster] = []
        indices = np.argsort(kmeans.cluster_centers_[cluster])[::-1]
        features = vec.get_feature_names()
        for i in indices[:n_words]:
            top_words[cluster].append(features[i])
    return top_words


top_words = get_top_words_per_cluster(df, vec, 3)
