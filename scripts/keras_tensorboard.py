"""
A simple script to experiment with tensorboard logging for keras entities
"""
import sys
import math
import logging

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from tensorflow.keras.callbacks import TensorBoard, Callback, LearningRateScheduler
import tensorflow.keras.backend as K

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
LOGFORMAT = '%(asctime)s | %(levelname)s | [%(filename)s:%(lineno)s - %(funcName)s] %(message)s'
logging.basicConfig(format=LOGFORMAT, stream=sys.stdout)


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class MyLearningRateScheduler(LearningRateScheduler):
    def __init__(self):
        super().__init__(self.step_decay)

    @staticmethod
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        logging.debug("Learning Rate change {}".format(lrate))
        return lrate


def main():
    x, y = make_classification()
    y = to_categorical(y)
    logging.debug((x.shape, y.shape))
    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_shape=[x.shape[1],]))
    model.add(keras.layers.Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tb = TensorBoard(log_dir='./logs')
    lr_logger = LRTensorBoard('.')
    my_learning_rate = MyLearningRateScheduler()
    model.fit(x, y, epochs=50, callbacks=[tb, my_learning_rate, lr_logger])


if __name__ == '__main__':
    main()