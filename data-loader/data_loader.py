"""
Create a data loading pipeline using tensorflow datasets
"""


from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


class BaseDataLoader(ABC):
    @abstractmethod
    def get_dataset(self) -> tf.data.Dataset:
        pass


class StandardDataLoader(BaseDataLoader):
    def __init__(self, flags):
        # self._flags = flags
        # self._csv_file = self._flags.csv_file
        self._df = pd.read_csv(flags)
        self._ds = None
        self._image_datagen = ImageDataGenerator()
        self._datagen = self._image_datagen.flow_from_dataframe(self._df, x_col='path', y_col='class_label',
                                                                target_size=(512, 743), shuffle=False)

    def make_generator(self):
        return self._datagen

    def get_dataset(self) -> tf.data.Dataset:
        n_channels = 3
        if self._datagen.color_mode == 'grayscale':
            n_channels = 1
        x_shape = [self._datagen.batch_size, ] + list(self._datagen.target_size) + [n_channels, ]
        y_shape = [self._datagen.batch_size, len(self._datagen.class_indices)]
        self._ds = tf.data.Dataset.from_generator(self.make_generator, output_types=(tf.float32, tf.float32),
                                                  output_shapes=(x_shape, y_shape))
        return self._ds

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
