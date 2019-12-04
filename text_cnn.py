from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np


# class ConvMaxPooling(layers.Layer):
#
#     def __init__(self, filters, max_words, embedded_length, words_covered, **kwargs):
#         kwargs['name'] = 'cov2D_max_pooling_concat'
#         super(ConvMaxPooling, self).__init__(**kwargs)
#
#         self.conv = layers.Conv2D(filters, (words_covered, embedded_length),
#                                   padding='same', activation='relu')
#
#         self.maxpooling = layers.MaxPooling2D(((max_words - words_covered + 1), 1))
#
#     def call(self, inp):
#         return self.maxpooling(self.conv(inp))


class TextCNN:

    def __init__(self, filters, filter_sizes, max_words, embedded_length, num_classes):
        self.filters = filters
        self.num_classes = num_classes

        input_layer = keras.Input(shape=(max_words, embedded_length, 1))

        if len(filter_sizes) == 0:
            raise TypeError("filter_sizes must be an iterable of int")

        if len(filter_sizes) > 1:
            concat_layers = []
            for words_covered in filter_sizes:
                conv = layers.Conv2D(filters, (words_covered, embedded_length),
                                     activation='relu')(input_layer)
                max_pooling = layers.MaxPooling2D(
                    ((max_words - words_covered + 1), 1))(conv)

                concat_layers.append(max_pooling)

            concat_layer = layers.concatenate(concat_layers)

        else:
            words_covered = filter_sizes[0]
            conv = layers.Conv2D(filters, (words_covered, embedded_length),
                                 padding='same', activation='relu')(input_layer)
            max_pooling = layers.MaxPooling2D(
                ((max_words - words_covered + 1), 1))(conv)

            concat_layer = max_pooling

        output_layer = layers.Dense(
            num_classes, activation='softmax')(concat_layer)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer)

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.CategoricalCrossentropy()
        )
        self.model.summary()

    def train_model(self, training_data, training_labels):

        print("training data shape:")
        print(training_data.shape)
        one_hot_targets = np.eye(self.num_classes)[training_labels]
        num_sample = len(training_labels)
        self.model.fit(training_data.reshape(training_data.shape + (1,)),
                       one_hot_targets.reshape(
                           (num_sample, 1, 1, self.num_classes)),
                       epochs=5)

    def eval(self, test_data, test_labels):

        print("Evaluating test set:")
        print(self.model.evaluate(test_data, test_labels))

    def predict(self, test_data):

        print("Prediction results:")
        print(self.model.predict(test_data))
