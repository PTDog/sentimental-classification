from tensorflow import keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np
import matplotlib.pyplot as plt


class TextCNN:

    def __init__(self, filters, filter_sizes, max_words, embedded_length, num_classes):
        self.filters = filters
        self.num_classes = num_classes
        self.history = None

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

        flatten_layer = layers.Flatten()(concat_layer)
        drop_out = layers.Dropout(0.2)(flatten_layer)
        output_layer = layers.Dense(
            num_classes, activation='softmax')(drop_out)

        self.model = keras.Model(inputs=input_layer,
                                 outputs=output_layer)

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        self.model.summary()

    def train(self, training_data, training_labels, test_data, test_labels):

        print("training data shape:")
        print(training_data.shape)

        training_tensor = training_data.reshape(training_data.shape + (1,))
        test_tensor = test_data.reshape(test_data.shape + (1,))

        training_labels_tensor = np.eye(self.num_classes)[training_labels] \
            .reshape((len(training_labels), 1, 1, self.num_classes))
        test_labels_tensor = np.eye(self.num_classes)[test_labels] \
            .reshape((len(test_labels), 1, 1, self.num_classes))

        self.history = self.model.fit(training_tensor, training_labels_tensor, epochs=5, batch_size=4000,
                                      validation_data=(test_tensor, test_labels_tensor))
        return self.history

    def eval(self, test_data, test_labels):

        print("Evaluating test set:")
        one_hot_targets = np.eye(self.num_classes)[test_labels]
        print(self.model.evaluate(test_data.reshape(test_data.shape + (1,)),
                                  one_hot_targets.reshape((len(test_labels), 1, 1, self.num_classes))))

    def predict(self, test_data):

        print("Prediction results:")
        print(self.model.predict(test_data.reshape(test_data.shape + (1,))))

    def plot_accuracy(self):

        history = self.history
        plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()


class TextNN:

    def __init__(self, max_words, embedded_length, num_classes):
        self.num_classes = num_classes
        self.history = None

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(max_words, embedded_length)))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        model.summary()
        self.model = model

    def train(self, training_data, training_labels, test_data, test_labels):
        print("training data shape:")
        print(training_data.shape)

        self.history = self.model.fit(training_data, training_labels, epochs=5, batch_size=4000,
                                      validation_data=(test_data, test_labels))
        return self.history


def plot_accuracy(histories, labels):

    plt.figure()
    for history, i in histories:
        label = labels[i]
        plt.plot(history.history['val_accuracy'], label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
