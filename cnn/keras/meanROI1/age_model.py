import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.noise import GaussianNoise


def build_model():
    model = Sequential()
    model.add(GaussianNoise(0.1, input_shape=(1,), name='age_noise1'))
    model.add(Dense(1, activation='relu', name='age_dense1', weights=[np.ones((1, 1), dtype=np.float32), np.array((0,))], trainable=False))
    return model
