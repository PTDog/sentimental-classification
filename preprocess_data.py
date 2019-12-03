import helper_function
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
# accept the directory of a file and transfer it to training set and testing set


def preprocess_data(path):
    # read data from path
    reviews = helper_function.read_data(path)

    # remove punctuation for all reviews
    reviews_with_no_punct = helper_function.remove_punct_mutiple(reviews)

    # tokenize all reviews
    reviews_tokens = helper_function.tokenize(reviews)

    # lower case all words
    reviews_lower_case = helper_function.lower_token_multiple(reviews_tokens)

    # remove stopwords
    reviews_no_stopwords = helper_function.remove_stopwords_mutiple(
        reviews_lower_case)

    all_tokens = []
    final_text = []
    for tokens in reviews_no_stopwords:
        all_tokens.append(tokens)
        final_text.append(' '.join(str(t) for t in tokens))

    data = defaultdict(list)
    data['final_text'] = final_text
    data['tokens'] = all_tokens
    all_data = pd.DataFrame(data)
    return all_data


def split_and_tokenize(data):
    data_train, data_test = train_test_split(
        data, test_size=0.10, random_state=42)
    TRAIN_VOCAB = helper_function.build_vocabulary(data_train)
    TEST_VOCAB = helper_function.build_vocabulary(data_test)
    cnn_data_train, embedding_weights_train = helper_function.get_word2vec_result(
        data_train, TRAIN_VOCAB, [], MAX_SEQUENCE_LENGTH=40, EMBEDDING_DIM=300)
    cnn_data_test, embedding_weights_test = helper_function.get_word2vec_result(
        data_test, TEST_VOCAB, [], MAX_SEQUENCE_LENGTH=40, EMBEDDING_DIM=300)
    return cnn_data_train, embedding_weights_train, cnn_data_test, embedding_weights_test
