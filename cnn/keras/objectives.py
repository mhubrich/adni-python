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


def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    MCC = numerator / (denominator + K.epsilon())
    MCC *= -1
    MCC += 1
    return K.tile(MCC, K.cast(K.sum(K.ones_like(y_true)), 'int32'))

