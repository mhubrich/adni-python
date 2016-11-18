from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.regularizers import l2
from keras.layers.noise import GaussianNoise


def build_model():
    model = Sequential()
    model.add(GaussianNoise(0.001, input_shape=(35,)))
    model.add(Dense(70, activation='relu', W_regularizer=l2(0.0001)))

    model.add(Dropout(0.15))
    model.add(Dense(70, activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dropout(0.15))
    model.add(Dense(35, activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dropout(0.15))
    model.add(Dense(16, activation='relu', W_regularizer=l2(0.0001)))
    model.add(Dropout(0.15))
    model.add(Dense(2, activation='softmax', W_regularizer=l2(0.0001)))

    return model
