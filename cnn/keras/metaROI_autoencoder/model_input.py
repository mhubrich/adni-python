from keras.layers import Input
from keras.models import Model


def build_model(input_shape=(128,)):
    return Input(shape=input_shape)

