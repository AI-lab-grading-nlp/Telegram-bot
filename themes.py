from chatbot import get_response, get_cheaper_response
from dotenv import load_dotenv, find_dotenv
from io import StringIO

# importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing sklearn libraries
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation

# importing deep learning libraries
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import gensim


def text_to_df(text: str) -> pd.DataFrame:
    # make a dataframe with 0 as a column name, splitting text into rows of 5 words and corresponding punctuation each
    df = pd.DataFrame(text.split('.'), columns=[0])
    return df


def embeddings(df: pd.DataFrame) -> np.ndarray:
    df[1] = df[0].apply(lambda x: gensim.utils.simple_preprocess(x))
    w2v_model = gensim.models.Word2Vec(
        df[1].values, window=5, min_count=2, vector_size=100)
    X = np.array([w2v_model.wv[word] for word in w2v_model.wv.index_to_key])

    return (X, w2v_model)


def topic_features(X, w2v_model):
    # Autoencoder
    input_dim = X.shape[1]
    encoding_dim = 100
    input_layer = Input(shape=(input_dim,))
    encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
    decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)
    autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X, X, epochs=50, batch_size=32)

    # Use encoder to extract features
    encoder = Model(inputs=input_layer, outputs=encoder_layer)
    features = encoder.predict(X)

    lda_featured = LatentDirichletAllocation(
        n_components=10, max_iter=50, learning_method='online', random_state=0)
    lda_featured.fit(features)

    # Get the feature names (i.e., the words in the vocabulary)
    feature_names = list(w2v_model.wv.index_to_key)

    # Get the topic-word distributions
    topic_word_distributions = lda_featured.components_

    # Get the indices of the top 10 words for each topic
    top_word_indices = topic_word_distributions.argsort(axis=1)[:, :-10:-1]

    # Convert the feature indices to feature names and store them in a list for each topic
    topic_features = []
    for topic_words in top_word_indices:
        feature_list = [feature_names[i] for i in topic_words]
        topic_features.append(feature_list)
    topic_features = list(set(topic_features))

    return topic_features[:4]


# def unify_top_words(top_words):
#     # create a list of all the top words, removing duplicates
#     top_words_list = []
#     for cluster in top_words:
#         for word in top_words[cluster]:
#             top_words_list.append(word)
#     top_words_list = list(set(top_words_list))
#     return top_words_list


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
