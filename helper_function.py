import re
import string
import glob
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# read data from files

stoplist = stopwords.words('english')
word2vec_path = '../GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def read_data(path):
    datafiles = glob.glob(path)
    reviews = []
    for file in datafiles:
        with open(file) as f:
            for line in f:
                reviews.append(line)
    return reviews


def remove_punct_single(review):
    res = re.sub('['+string.punctuation+']', '', review)
    return res


def remove_punct_mutiple(reviews):
    res = list(map(remove_punct_single, reviews))
    return res


def tokenize(reviews):
    res = list(map(word_tokenize, reviews))
    return res


def lower_token_single(tokens):
    return [t.lower() for t in tokens]


def lower_token_multiple(tokens):
    return list(map(lower_token_single, tokens))


def remove_stopwords_single(token):
    return [word for word in token if word not in stoplist]


def remove_stopwords_mutiple(tokens):
    return list(map(remove_stopwords_single, tokens))


# helper functions for split and tokenize
def build_vocabulary(data):
    all_words = [word for tokens in data['tokens'] for word in tokens]
    review_lengths = [len(tokens) for tokens in data['tokens']]
    VOCAB = sorted(list(set(all_words)))
    return VOCAB


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:  # this list is empty
        return np.zeros(k)
    if generate_missing:  # padding with random list
        vectorized = [vector[word] if word in vector else np.random.rand(
            k) for word in tokens_list]
    else:  # padding with zero sequence
        vectorized = [vector[word] if word in vector else np.zeros(
            k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, tokens, generate_missing=False):
    embeddings = []
    for x in tokens:
        embeddings.append(get_average_word2vec(
            x, vectors, generate_missing=generate_missing))
    return embeddings


def get_word2vec_result(data, VOCAB, tokens, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    # training_embeddings = get_word2vec_embeddings(
        # word2vec, tokens, generate_missing=True)

    # # Tokenize and padding sequence (Training data)
    tokenizer = Tokenizer(num_words=len(VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data['final_text'].tolist())
    sequence = tokenizer.texts_to_sequences(
        data['final_text'].tolist())

    data_word_index = tokenizer.word_index

    cnn_data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    embedding_weights = np.zeros((len(data_word_index) + 1, EMBEDDING_DIM))
    for word, index in data_word_index.items():
        embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(
            EMBEDDING_DIM)
    return cnn_data, embedding_weights


def combine_data_and_weights(data, weights):
    max_review_length = len(data[0])
    number_of_reviews = len(data)
    newlist = []
    for review in data:
        for word_index in review:
            newlist.append(weights[word_index - 1])
    res = np.reshape(np.array(newlist),
                     (number_of_reviews, max_review_length, 300))
    return res
