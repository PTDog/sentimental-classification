from text_cnn import TextCNN
# from main import get_data

import numpy as np
NUM_FILTERS = 3
FILTER_SIZES = [2, 3, 5]
MAX_WORD = 40
EMBEDDING_LENGTH = 300
NUM_CLASSES = 2

model = TextCNN(NUM_FILTERS, FILTER_SIZES, MAX_WORD,
                EMBEDDING_LENGTH, NUM_CLASSES)
# pos_path = "./review_polarity/txt_sentoken/pos/*.txt"
# neg_path = "./review_polarity/txt_sentoken/neg/*.txt"
# pos_data_train, pos_data_test = get_data(pos_path)
# neg_data_train, neg_data_test = get_data(neg_path)
#
#
# def read_data_from_csv(path):
#     res = []
#     with open(path, 'rb') as f:
#         review = []
#         for line in f:
#             nums = line.split(',')
#             newWord = [float(num) for num in nums]
#             review.append(newWord)
#             if len(review) == 40:
#                 res.append(review)
#                 review = []
#     return np.array(res)
#
#
# pos_data_train = read_data_from_csv("./training_training_data_pos.csv")
# pos_data_test = read_data_from_csv("./testing_data_pos.csv")
# neg_data_train = read_data_from_csv("./training_training_data_neg.csv")
# neg_data_test = read_data_from_csv("./testing_data_neg.csv")
# print(pos_data_test)
#
# pos_labels_train = np.ones(len(pos_data_train), dtype=int)
# neg_labels_train = np.zeros(len(neg_data_train), dtype=int)
#
# train_data = np.concatenate((pos_data_train, neg_data_train))
# train_label = np.concatenate((pos_labels_train, neg_labels_train))
#
# pos_labels_test = np.ones(len(pos_data_test), dtype=int)
# neg_labels_test = np.zeros(len(neg_data_test), dtype=int)
#
# test_data = np.concatenate((pos_data_test, neg_data_test))
# test_label = np.concatenate((pos_labels_test, neg_labels_test))
#
#
# model.train(train_data, train_label, test_data, test_label)
# model.plot_accuracy()
# model.eval(test_data, test_label)
# model.predict(test_data)
