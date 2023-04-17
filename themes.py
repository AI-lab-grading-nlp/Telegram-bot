from chatbot import get_response, get_cheaper_response
from dotenv import load_dotenv, find_dotenv
from io import StringIO
import openai
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

pd.set_option('display.max_colwidth', 100)


# df = pd.read_csv(source, sep='.')  # add data here

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
                          norm='l2', max_df=0.95)

    X_vec = vec.fit_transform(df)  # fitting and transforming the data

    scaler = StandardScaler(with_mean=False)  # scaling the data
    X_fit = scaler.fit(X_vec)
    X = scaler.transform(X_vec)

    return (X, vec)


def get_top_words_per_cluster(df, vec, n_words, kmeans):
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
        features = vec.get_feature_names_out()
        for i in indices[:n_words]:
            top_words[cluster].append(features[i])
    return top_words


def create_clusters(df, clusters: int = 5):
    """
    Creates clusters using the KMeans algorithm.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the text data.

    Returns:
        df (pd.DataFrame): A pandas DataFrame containing the text data and cluster labels.
    """
    df.rename(columns={0: 'corpus'}, inplace=True)
    df['cleaned'] = df['corpus'].apply(
        lambda x: clean_data(x, remove_stopwords=True))
    X, vec = vectorize_tfidf(df['cleaned'])
    kmeans = KMeans(n_clusters=clusters, n_init="auto", random_state=0).fit(X)
    df['cluster'] = kmeans.labels_
    top_words = get_top_words_per_cluster(df, vec, 3, kmeans)
    return df, top_words


def text_to_df(text: str) -> pd.DataFrame:
    """
    Converts a string of text into a pandas DataFrame.

    Args:
        text (str): A string of text to be converted into a DataFrame.

    Returns:
        df (pd.DataFrame): A pandas DataFrame containing the text data.
    """
    # make a dataframe with 0 as a column name, splitting text into rows of 5 words and corresponding punctuation each
    df = pd.DataFrame(text.split('.'), columns=[0])
    return df


def unify_top_words(top_words):
    # create a list of all the top words, removing duplicates
    top_words_list = []
    for cluster in top_words:
        for word in top_words[cluster]:
            top_words_list.append(word)
    top_words_list = list(set(top_words_list))
    return top_words_list


# def themes_pipeline(text: str, clusters: int = 5) -> pd.DataFrame:
#     """
#     Creates clusters using the KMeans algorithm.

#     Args:
#         df (pd.DataFrame): A pandas DataFrame containing the text data.

#     Returns:
#         top_words_list (list): A list of the top words for all clusters to be used in themes.
#     """
#     df = text_to_df(text)
#     df, top_words = create_clusters(df, clusters=clusters)
#     top_words_list = unify_top_words(top_words)
#     min_length = min(len(top_words_list), 10)
#     return top_words_list[:min_length]


text = """
/source Principal Components Analysis (PCA) is a well-known unsupervised dimensionality reduction technique that constructs relevant features/variables through linear (linear PCA) or non-linear (kernel PCA) combinations of the original variables (features). In this post, we will only focus on the famous and widely used linear PCA method.

The construction of relevant features is achieved by linearly transforming correlated variables into a smaller number of uncorrelated variables. This is done by projecting (dot product) the original data into the reduced PCA space using the eigenvectors of the covariance/correlation matrix aka the principal components (PCs).

The resulting projected data are essentially linear combinations of the original data capturing most of the variance in the data (Jolliffe 2002).

In summary, PCA is an orthogonal transformation of the data into a series of uncorrelated data living in the reduced PCA space such that the first component explains the most variance in the data with each subsequent component explaining less.
"""


load_dotenv()

def clean_themes(themes: list[str]) -> list[str]:

    new_themes = []
    for t in themes:
        t = clean_data(t, remove_stopwords=True)
        new_themes.append(t)

    return new_themes


def themes_pipeline(text: str, cheaper=True) -> list[str]:
    '''
    Placeholder for theme extraction that uses OpenAI's API to extract themes from a given text.
    '''

    if cheaper:
        messages = [{'role': 'user', 'content': text}, {'role': 'system',
                                                        'content': 'What are the themes of this text? separate each theme with a comma.'}]
        response = get_cheaper_response(messages)

    else:
        pre = 'System: You are a helpful chatbot. \n\nUser:'
        message = '\n\nProvide a list of themes for the given text, separate each theme with a comma.\n\nBot:'
        response = get_response(pre + text + message)

    themes = response.split(',')

    print('old themes', themes)
    # clean themes
    themes = clean_themes(themes)
    print('new themes', themes)

    return themes
