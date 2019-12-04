from main import get_data
import csv
import numpy as np
pos_path = "./review_polarity/txt_sentoken/pos/*.txt"
neg_path = "./review_polarity/txt_sentoken/neg/*.txt"
pos_data_train, pos_data_test = get_data(pos_path)
neg_data_train, neg_data_test = get_data(neg_path)
print("shape of pos data train")
print(pos_data_train.shape)
print("shape of pos data test")
print(pos_data_test.shape)
print("shape of neg data train")
print(neg_data_train.shape)
print("shape of neg data test")
print(neg_data_test.shape)
with open('training_data_pos.csv', 'w',) as file:
    writer = csv.writer(file)
    for review in pos_data_train:
        for word in review:
            writer.writerow(word)
with open('testing_data_pos.csv', 'w',) as file:
    writer = csv.writer(file)
    for review in pos_data_test:
        for word in review:
            writer.writerow(word)

with open('testing_data_neg.csv', 'w',) as file:
    writer = csv.writer(file)
    for review in neg_data_test:
        for word in review:
            writer.writerow(word)

with open('training_data_neg.csv', 'w') as file:
    writer = csv.writer(file)
    for review in neg_data_train:
        for word in review:
            writer.writerow(word)
