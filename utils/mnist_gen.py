from __future__ import absolute_import, division

import tensorflow as tf
from tensorflow.python import keras


def get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def get_gen(set_name, batch_size, shuffle=True, scaled=False):
    translate = 1.0 if scaled else 0.0
    scale = (1.0, 2.5) if scaled else (1.0, 1.0)
    if set_name == 'train':
        (X, Y), _ = get_mnist_dataset()
    elif set_name == 'test':
        _, (X, Y) = get_mnist_dataset()

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=scale,
        width_shift_range=translate,
        height_shift_range=translate)
    gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen
