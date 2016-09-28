from keras import backend as K

# Number of unlabeled instances inside a bag.
# Typically, this corresponds to the number of
# slices we extract from a single scan.
INSTANCES = 64


def mil_squared_error(y_true, y_pred):
    # TODO At the moment, only class_mode = 'binary' is supported
    return K.tile(K.square(K.max(y_pred) - y_true[0]), INSTANCES)
    # return K.tile(K.square(K.max(y_pred) - K.max(y_true)), K.count_params(y_true))
    # return K.square(K.max(y_pred, axis=-1) - K.mean(y_true, axis=-1))
    # Normal: return K.mean(K.square(y_pred - y_true), axis=-1)
