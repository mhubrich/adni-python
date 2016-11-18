from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.noise import GaussianNoise


def build_model():
    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(5,)))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    return model
