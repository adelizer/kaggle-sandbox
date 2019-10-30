"""
Create a data loading pipeline using tensorflow datasets
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


class DataLoader:
    def __init__(self):
        x = np.ones((100,4))
        y = np.ones((100,1))
        self._dataset = tf.data.Dataset.from_tensor_slices((x, to_categorical(y)))

    def get_dataset(self):
        self._dataset = self._dataset.batch(16).repeat()
        return self._dataset


def main():
    d = DataLoader()

    model = keras.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(4,), activation='tanh'))
    model.add(keras.layers.Dense(16, activation='tanh'))
    model.add(keras.layers.Dense(2, activation='tanh'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    dataset = d.get_dataset()
    model.fit(dataset, epochs=2, steps_per_epoch=10)


if __name__ == '__main__':
    main()