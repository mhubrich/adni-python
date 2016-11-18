from keras.layers import Input, Convolution3D, Flatten


def build_model(input_shape=(1, 29, 29, 29)):
    input = Input(shape=input_shape)

    x = Convolution3D(32, 5, 5, 5, activation='relu')(input)
    x = Convolution3D(32, 5, 5, 5, activation='relu')(x)
    x = Convolution3D(32, 5, 5, 5, activation='relu')(x)

    x = Convolution3D(64, 3, 3, 3, activation='relu')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu')(x)

    x = Convolution3D(128, 3, 3, 3, activation='relu')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu')(x)

    x = Flatten()(x)

    return input, x
