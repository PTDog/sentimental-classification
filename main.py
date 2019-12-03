import preprocess_data
import numpy as np
from helper_function import combine_data_and_weights
pos_path = "./review_polarity/txt_sentoken/pos/cv000_29590.txt"
neg_path = "./review_polarity/txt_sentoken/neg/cv000_29416.txt"

pos_reviews = preprocess_data.preprocess_data(pos_path)

pos_cnn_data_train, pos_embedding_weights_train, pos_cnn_data_test, pos_embedding_weights_test = preprocess_data.split_and_tokenize(
    pos_reviews)

pos_data_train = combine_data_and_weights(
    pos_cnn_data_train, pos_embedding_weights_train)
pos_data_test = combine_data_and_weights(
    pos_cnn_data_test, pos_embedding_weights_test)

print(pos_data_test)
print(pos_data_train)
