from keras.backend import epsilon
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import log_loss, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, np.round(y_pred))


def loss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def TP(y_true, y_pred):
    # TP = tp/(tp+fn)
    conf = confusion_matrix(y_true, np.round(y_pred))
    return float(conf[1,1]) / (conf[1,1] + conf[1,0] + epsilon())


def TN(y_true, y_pred):
    # TN = (tn/tn+fp)
    conf = confusion_matrix(y_true, np.round(y_pred))
    return float(conf[0,0]) / (conf[0,0] + conf[0,1] + epsilon())


def mean_accuracy(y_true, y_pred):
    # MEAN ACC = (TP + TN) / 2
    return (TP(y_true, y_pred) + TN(y_true, y_pred)) / 2.0


def fmeasure(y_true, y_pred):
    if np.sum(np.round(y_pred)) == 0:
        return 0
    else:
        return f1_score(y_true, np.round(y_pred))


def matthews_correlation(y_true, y_pred):
    return matthews_corrcoef(y_true, np.round(y_pred))


class Evaluation(Callback):
    def __init__(self, x_val, y_val, batch_size=32, callbacks=[]):
        super(Evaluation, self).__init__()
        self.X = x_val
        self.y_true = y_val
        self.batch_size=batch_size
        self.callbacks = callbacks

    def set_params(self, params):
        super(Evaluation, self).set_params(params)
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        super(Evaluation, self).set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X, self.batch_size, verbose=0)
        # In case of multi-output-model use only first output
        if isinstance(pred, list):
            pred = pred[0]
        logs['val_loss'] = loss(self.y_true, pred)
        logs['val_acc'] = accuracy(self.y_true, pred)
        logs['val_fmeasure'] = fmeasure(self.y_true, pred)
        logs['val_mean_acc'] = mean_accuracy(self.y_true, pred)
        logs['val_mcc'] = matthews_correlation(self.y_true, pred)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_epoch_begin(self, epoch, logs={}):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_batch_begin(self, batch, logs={}):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs={}):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs={}):
        for callback in self.callbacks:
            callback.on_train_end(logs)


