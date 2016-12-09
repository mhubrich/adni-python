from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.noise import GaussianNoise


def build_model(num_features=5):
    model = Sequential()
    model.add(GaussianNoise(0.001, input_shape=(num_features,), name='diff_noise1'))
    model.add(Dense(5, activation='relu', name='diff_dense1'))
    model.add(Dropout(0.1, name='diff_dropout1'))
    model.add(Dense(5, activation='relu', name='diff_dense2'))
    model.add(Dropout(0.1, name='diff_dropout2'))
    model.add(Dense(5, activation='relu', name='diff_dense3'))
    model.add(Dropout(0.1, name='diff_dropout3'))
    model.add(Dense(5, activation='relu', name='diff_dense4'))
    model.add(Dropout(0.1, name='diff_dropout4'))
    model.add(Dense(5, activation='relu', name='diff_dense5'))
    model.add(Dropout(0.1, name='diff_dropout5'))
    model.add(Dense(1, activation='sigmoid', name='diff_dense6'))
    return model
