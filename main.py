import preprocess_data
import numpy as np
from helper_function import combine_data_and_weights


def get_data(path):
    reviews = preprocess_data.preprocess_data(path)
    cnn_data_train, embedding_weights_train, cnn_data_test, embedding_weights_test = preprocess_data.split_and_tokenize(
        reviews)
    pos_data_train = combine_data_and_weights(
        cnn_data_train, embedding_weights_train)
    pos_data_test = combine_data_and_weights(
        cnn_data_test, embedding_weights_test)
    return pos_data_train, pos_data_test
