"""
Create a data loading pipeline using tensorflow datasets
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class DataLoader:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        one_hot_y_train = to_categorical(y_train)
        one_hot_y_test = to_categorical(y_test)
        self.image_gen = ImageDataGenerator(rotation_range=50, shear_range=10)
        self.datagen = self.image_gen.flow(x=x_train, y=one_hot_y_train, batch_size=32, shuffle=True, save_to_dir='.')

    def get_dataset(self):
        self._dataset = self._dataset.batch(32).repeat()
        return self._dataset


def main():
    d = DataLoader()
    print("Number of samples: ", d.datagen.n)
    d.datagen.next()
    model = keras.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(4,), activation='tanh'))
    model.add(keras.layers.Dense(16, activation='tanh'))
    model.add(keras.layers.Dense(2, activation='tanh'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    dataset = d.get_dataset()
    model.fit(dataset, epochs=2, steps_per_epoch=10)


if __name__ == '__main__':
    main()