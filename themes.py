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

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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

    topic_features = []
    # Convert the feature indices to feature names and store them in a list for each topic
    for topic_words in top_word_indices:

        try:
            feature_list = [feature_names[i] for i in topic_words]
        except:
            continue
        topic_features.append(feature_list)

    topic_features = [set(x) for x in topic_features] # list of sets
    # flatten the list of sets
    topic_features = [item for sublist in topic_features for item in sublist]
    # remove duplicates
    topic_features = list(set(topic_features))

    return topic_features


# def unify_top_words(top_words):
#     # create a list of all the top words, removing duplicates
#     top_words_list = []
#     for cluster in top_words:
#         for word in top_words[cluster]:
#             top_words_list.append(word)
#     top_words_list = list(set(top_words_list))
#     return top_words_list


text = """
Summary
This short essay describes a methodology for program evaluation known as
encouragement design that is particularly useful when one wishes to rigorously
estimate the effect of some intervention (such as accessing financing or simplified
business registration) that cannot itself be randomly administered to some and not
others. The method is called an encouragement design because it requires that a
randomly-selected group of beneficiaries receive extra encouragement to undertake
the intervention. Encouragement typically takes the form of additional information
or incentives. A successful encouragement design allows estimation of the effect of
the intervention as well as the effect of the encouragement itself. IFC is currently
planning an encouragement design in an Alternate Dispute Resolution (Business
Enabling Environment) project in Macedonia.
Experimental Design
A randomized experiment is a study designed to estimate the effect of an
intervention on a set of indicators (“results”) for a collection of units (e.g., individuals
or firms). In a randomized experiment, the researcher can manipulate the exposed
group (the “treatment group”) and the not-exposed group (the “control group”).
Randomization is useful because it ensures that (on average), prior to the
intervention, treatment and control groups are essentially identical and therefore
would show very similar results in the absence of the treatment. Therefore, a
difference in results for the two groups can be causally attributed to the program.
The Encouragement Design
In a classic experiment, the researcher has full control over which subjects will be
exposed to an intervention. In practice, experiments can be difficult to implement
because it is difficult to ensure that all those selected for treatment get the
treatment and all selected for control do not. Also, it is sometimes impractical or
unethical to conduct experiments and force some of the subjects to be in the control
group. For example, in an access-to-finance program, we would like to know the
effect of receiving a loan on firm-level impacts, but we cannot randomize loan
approvals. Similarly, in a business-enabling-environment program we would like to
estimate the effect of registering a business on firm-level impacts, but we cannot
force some firms to become registered and other firms to remain informal.
The encouragement design is a special case of an experimental design that can be
used in situations with little control over subjects’ compliance. The key idea is that
instead of randomizing the application of the intervention itself, what is randomized
is encouragement to receive the treatment. By randomizing encouragement and
carefully tracking outcomes for all those who do and do not receive the
encouragement, it is possible to obtain reliable estimates of BOTH the
encouragement and the intervention itself (see “Quantities of Interest”, below).
Encouragement may take the form of information that is additional to whatever is
already part of program implementation and targeted at the unit (e.g., firm) level.
For example, in the context of a business-registration simplification program, the
project team may plan to advertise the simplification on TV and radio; the

encouragement could take the form of additional direct mailings to a random sample

of firms. In the case of access-to-finance, encouragement can take the form of loan-
application training—the advantage here is that one can then estimate the effect of

training on receiving a loan as well as the effect of the loan on firm-level impacts.
Note that encouragement is merely that—encouragement. Some firms receiving
encouragement may not follow through with their loan or business registration
application. And some firms who do not receive encouragement may get loans or
register their businesses. In order for the “encouragement design” to work, the
encouragement must significantly increase the likelihood that units will follow
through with what they are being encouraged to do.
Quantities of Interest
A. The effect of encouragement
If encouragement (e.g., information, training, etc.) is randomly applied to some
units and not others, then estimating the effect of encouragement on outcomes is
simple: compare outcomes for the randomly-selected encouraged group vs.
outcomes for the randomly-selected not-encouraged group. This quantity of
interest, known as the “Intention-to-Treat” effect, or ITT, is the effect of the
encouragement on outcomes, e.g., “What is the effect of direct-mail advertising on
business performance?”
Since encouragement itself is randomized, the comparison between the encouraged

(e.g., trained) and not-encouraged groups will be free of any bias due to self-
selection. What is important here is that one can measure results for all subjects

from the sample in which randomized selection was undertaken. Whenever
encouragement takes the form of key project activities (e.g., training, or firm-level
informational campaigns), measuring the impact of encouragement is particularly
useful.
B. The effect of the treatment
In many situations we are also interested in the effect of the larger event or
intervention (the “treatment”, e.g., obtaining the financing, registering the business)
on firm-level impacts. We wish to answer questions such as: "What is the effect of
accessing finance on employment and revenues?" This effect is often difficult to
estimate because up-take of treatment itself is not randomized and therefore we
cannot simply compare results for treated vs. untreated firms.
However, one can still estimate the effect of the treatment by exploiting a

randomized encouragement design, adjusting the ITT effect by the amount of non-
compliance. This yields the local average treatment effect (LATE), computed as:

LATE = ITT / Compliance Rate
where Compliance Rate = Fraction of Subjects that were treated in the encouraged
group - Fraction of Subjects that were treated in the non-encouraged group.
If the compliance rate is 100%, LATE = ITT, we have perfect compliance, all
assigned to the treatment take the treatment and all those assigned to the control
do not take the treatment. The compliance rate can be thought of as the fraction of
subjects that fall into the sub-population of “compliers”, the group for whom the
decision to take treatment was directly affected by the assignment. Put differently,
this is the group induced by the encouragement to take advantage of the treatment.

Notice that the compliers can be thought of as the group of people that actually stick
to the experimental protocol; they will take the treatment if assigned to the
treatment group, but they will not take the treatment if assigned to the control

group. From the point-of-view of a policymaker, compliers are an interesting sub-
population because they are the only ones who are actually affected by the

encouragement. Note that not all subjects in the sample will be compliers: some will
always follow through with treatment, and some will never take the treatment
regardless of their assignment.
Usually, the compliance rate will be less then one, and then it becomes important to

recognize that the LATE effect estimates the effect of treatment only for the sub-
population of compliers and it does not constitute the effect of the treatment for the

whole sample.
An important special case is when the control group can be excluded from taking the
treatment. Then non-compliance can only occur in the treatment group and the LATE
will be equal to the average treatment effect for the treated—that is, the average
effect of the treatment for those that do take the treatment. In general, the
compliance rate depends on the encouragement. Some encouragements will be
relatively effective and strongly influence take up. Other encouragements may be
less effective.
There are a few important assumptions that need to hold for the LATE to give an
unbiased estimate.
 Encouragement cannot make subjects less likely to receive the treatment.
This is often a reasonable assumption, but needs to be considered carefully
on a case-by-case basis.
 Encouragement is in fact randomly assigned (or is “as good as random”);
those assigned to encouragement receive it, and the rest do not.
 Encouragement has no direct effect on results, except via increasing the
probability of receiving treatment. This is why encouragement should be kept
as simple as possible. For example, if encouragement takes the form of a
training session that could have its own effect on results in addition to
encouraging take-up of treatment, then this assumption would be violated.
Such a case would be a training session that combines lessons on accessing
finance and successful business management."""


def abeera_pipeline(text: str) -> list[str]:
    df = text_to_df(text)
    X, w2v_model = embeddings(df)
    top_words = topic_features(X, w2v_model)
    return top_words



load_dotenv()

stopwords.words('english')
import re, math


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

def clean_themes(themes: list[str]) -> list[str]:

    new_themes = []
    for t in themes:
        t = clean_data(t, remove_stopwords=True)
        new_themes.append(t)

    return new_themes


def themes_pipeline(text: str, cheaper=True, abeera=False) -> list[str]:
    '''
    Placeholder for theme extraction that uses OpenAI's API to extract themes from a given text.
    '''

    if cheaper:
        messages = [{'role': 'user', 'content': text}, {'role': 'system',
                                                        'content': 'What are the themes of this text? separate each theme with a comma.'}]
        response = get_cheaper_response(messages)

    elif abeera:
        response = abeera_pipeline(text)

    else:
        pre = 'System: You are a helpful chatbot. \n\nUser:'
        message = '\n\nProvide a list of themes for the given text, separate each theme with a comma.\n\nBot:'
        response = get_response(pre + text + message)

    if not abeera:
        themes = response.split(',')
    else:
        themes = response

    print('old themes', themes)
    # clean themes
    themes = clean_themes(themes)
    print('new themes', themes)

    return themes
print(themes_pipeline(text, cheaper=False, abeera=True))
