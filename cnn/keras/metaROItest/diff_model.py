from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.noise import GaussianNoise


def build_model():
    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=(5,), name='diff_noise1'))
    model.add(Dense(8, activation='relu', name='diff_dense1'))
    model.add(Dropout(0.1, name='diff_dropout1'))
    model.add(Dense(8, activation='relu', name='diff_dense2'))
    model.add(Dropout(0.1, name='diff_dropout2'))
    model.add(Dense(8, activation='relu', name='diff_dense3'))
    model.add(Dropout(0.1, name='diff_dropout3'))
    #model.add(Dense(2, activation='softmax', name='diff_dense4'))
    return model
