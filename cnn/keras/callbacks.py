"""
For further information, see: http://keras.io/callbacks/
"""
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
import os
import time
import sys
import warnings
import numpy as np

VERBOSITY = 1


def checkpoint(path_dir):
    # Make sure directory exists
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    return ModelCheckpoint(
        os.path.join(path_dir, 'weights.{epoch:02d}-loss_{loss:.3f}-acc_{acc:.3f}.h5'),
        verbose=VERBOSITY,
        save_best_only=False)


class _OptimizerSaver(Callback):
    def __init__(self, optimizer, path_dir, save_only_last=False, verbose=1):
        super(_OptimizerSaver, self).__init__()
        assert hasattr(optimizer, 'save_weights'), \
            'Optimizer "%s" has no method save_weights().' % optimizer.__class__.__name__
        assert hasattr(optimizer, 'save_updates'), \
            'Optimizer "%s" has no method save_updates().' % optimizer.__class__.__name__
        assert hasattr(optimizer, 'save_config'), \
            'Optimizer "%s" has no method save_config().' % optimizer.__class__.__name__
        self.optimizer = optimizer
        self.path_dir = path_dir
        self.save_only_last = save_only_last
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        optimizer_name = self.optimizer.__class__.__name__
        filepath_weights = ''
        filepath_updates = ''
        filepath_config = ''
        if not self.save_only_last:
            filepath_weights = '.{epoch:02d}'
            filepath_updates = '.{epoch:02d}'
            filepath_config = '.{epoch:02d}'
        filepath_weights = optimizer_name + '_weights' + filepath_weights + '.p'
        filepath_updates = optimizer_name + '_updates' + filepath_updates + '.p'
        filepath_config = optimizer_name + '_config' + filepath_config + '.p'
        filepath_weights = filepath_weights.format(epoch=epoch)
        filepath_updates = filepath_updates.format(epoch=epoch)
        filepath_config = filepath_config.format(epoch=epoch)
        filepath_weights = os.path.join(self.path_dir, filepath_weights)
        filepath_updates = os.path.join(self.path_dir, filepath_updates)
        filepath_config = os.path.join(self.path_dir, filepath_config)
        self.optimizer.save_weights(filepath_weights)
        self.optimizer.save_updates(filepath_updates)
        self.optimizer.save_config(filepath_config)
        if self.verbose:
            print('Saved weights of optimizer: %s' % filepath_weights)
            print('Saved updates of optimizer: %s' % filepath_updates)
            print('Saved config of optimizer: %s' % filepath_config)


def save_optimizer(optimizer, path_dir, save_only_last=False):
    # Make sure directory exists
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    return _OptimizerSaver(optimizer, path_dir, save_only_last=save_only_last, verbose=VERBOSITY)


class _LearningRate:
    def __init__(self, lr, decay_rate, decay_epochs):
        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs

    def decayed_learning_rate(self, epoch):
        lr = self.lr * pow(self.decay_rate, epoch / self.decay_epochs)  # Note, this is an integer division
        if VERBOSITY:
            print('Learning rate in epoch %d: %f' % (epoch+1, lr))
        return lr


def learning_rate(lr, decay_rate, decay_epochs):
    lr = _LearningRate(lr, decay_rate, decay_epochs)
    return LearningRateScheduler(lr.decayed_learning_rate)


def tensorboard(path_log_dir):
    return TensorBoard(
        log_dir=path_log_dir,
        histogram_freq=1,
        write_graph=True)


class _BatchLogger(Callback):
    """
    Custom Callback.
    Prints the duration of each batch during training.
    """
    def __init__(self, step, verbose=1):
        super(_BatchLogger, self).__init__()
        self.step = step
        self.verbose = verbose
        self.num_batch = 0
        self.start = 0

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            self.num_batch = 0
            self.start = time.time()

    def on_batch_end(self, epoch, logs={}):
        self.num_batch += 1
        if self.verbose and self.num_batch % self.step == 0:
            duration = time.time() - self.start
            daytime = time.strftime('%H:%M:%S: ')
            output = daytime + 'Batch %04d: %.2f Examples/Second, %.2f Seconds/Batch'
            o1 = (self.step * logs.get('size', 0)) / duration
            o2 = duration / self.step
            print(output % (self.num_batch, o1, o2))
            sys.stdout.flush()
            self.start = time.time()


def batch_logger(step):
    return _BatchLogger(step, VERBOSITY)


class _HistoryPrinter(Callback):
    def __init__(self, verbose=1):
        super(_HistoryPrinter, self).__init__()
        self.verbose = verbose
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs['acc'])
        self.loss.append(logs['loss'])
        if 'val_acc' in logs:
            self.val_acc.append(logs['val_acc'])
            self.val_loss.append(logs['val_loss'])

    def on_train_end(self, logs={}):
        if self.verbose:
            print('\n')
            for i in range(0, len(self.acc)):
                if len(self.val_acc) > 0:
                    s = 'Epoch %d -- Acc: %.5f, Loss: %.5f -- Val Acc: %.5f, Val Loss: %.5f'
                    print(s % (i+1, self.acc[i], self.loss[i], self.val_acc[i], self.val_loss[i]))
                else:
                    s = 'Epoch %d -- Acc: %.5f, Loss: %.5f'
                    print(s % (i + 1, self.acc[i], self.loss[i]))


def print_history():
    return _HistoryPrinter(VERBOSITY)


class _MyModelCheckpoint(Callback):
    '''Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.
    # Arguments
        filepath: string, path to save the model file.
        monitor: list of quantities to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the 'max_files' latest best models according to
            the quantity monitored will not be overwritten.
        max_files: if `save_best_only=True`, then only 'max_files' files
            per quantity will be kept.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
    '''
    def __init__(self, filepath, monitor=['val_loss', 'val_acc', 'val_fmeasure'], verbose=0,
                 save_best_only=True, max_files=5, save_weights_only=False):
        super(_MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.max_files = max_files
        self.save_weights_only = save_weights_only
        self.monitor_op = []
        self.reverse = []
        self.best = np.zeros((len(self.monitor), self.max_files), dtype=np.float64)
        self.files = np.empty((len(self.monitor), self.max_files), dtype=np.object)
        for i in range(len(self.monitor)):
            if 'loss' in self.monitor[i]:
                self.monitor_op.append(np.less)
                self.best[i].fill(np.Inf)
                self.reverse.append(False)
            else:
                self.monitor_op.append(np.greater)
                self.best[i].fill(-np.Inf)
                self.reverse.append(True)

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            for i in range(len(self.monitor)):
                current = logs.get(self.monitor[i])
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor[i]), RuntimeWarning)
                else:
                    if self.monitor_op[i](current, self.best[i][-1]):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor[i], self.best[i][-1],
                                     current, filepath))
                        if len(np.extract(self.files == filepath, self.files)) == 0:
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        if self.files[i][-1] is not None and len(np.extract(self.files == self.files[i][-1], self.files)) < 2:
                            os.remove(self.files[i][-1])
                        self.best[i][-1] = current
                        self.files[i][-1] = filepath
                        indices = np.argsort(self.best[i])
                        if self.reverse[i]:
                            indices = indices[::-1]
                        self.best[i] = self.best[i][indices]
                        self.files[i] = self.files[i][indices]
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor[i]))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)


def save_model(path_dir, monitor=['val_loss', 'val_acc', 'val_fmeasure'], verbose=0,
               save_best_only=True, max_files=5, save_weights_only=False):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    fname = 'model.{epoch:04d}-loss_{loss:.3f}-acc_{acc:.3f}'
    for m in monitor:
        fname += '-' + m + '_{' + m + ':.4f}'
    fname += '.h5'
    return _MyModelCheckpoint(os.path.join(path_dir, fname),
                              monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                              max_files=max_files, save_weights_only=save_weights_only)


class _Flush(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        sys.stdout.flush()


def flush():
    return _Flush()


class _TrainAccStopping(Callback):
    '''Stop training when trainings accuracy is greater than a threshold.
    # Arguments
        max_acc: Threshold to stop training.
        patience: number of epochs with trainings accuracy greater than
            'max_acc' after which training will be stopped.
        verbose: verbosity mode.
    '''
    def __init__(self, max_acc=0.95, patience=0, verbose=0):
        super(_TrainAccStopping, self).__init__()
        self.monitor = 'acc'
        self.monitor_op = np.greater
        self.max_acc = max_acc
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.max_acc):
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1
        else:
            self.wait = 0

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


def early_stopping(max_acc=0.95, patience=5, verbose=VERBOSITY):
    return _TrainAccStopping(max_acc=max_acc,
                             patience=patience,
                             verbose=verbose)


class EarlyStop(Callback):
    '''Stop training when all monitored quantities have stopped improving.
    # Arguments
        monitor: quantities to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
    '''
    def __init__(self, monitor=['val_loss', 'val_acc'], patience=0, verbose=0):
        super(EarlyStop, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.ops = []
        for m in self.monitor:
            if 'loss' in m:
                self.ops.append(np.less)
            else:
                self.ops.append(np.greater)

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = []
        for i in range(len(self.ops)):
            self.best.append(np.Inf if self.ops[i] == np.less else -np.Inf)

    def on_epoch_end(self, epoch, logs={}):
        flag = False
        for i in range(len(self.monitor)):
            current = logs.get(self.monitor[i])
            if current is None:
                warnings.warn('Early stopping requires %s available!' %
                              (self.monitor[i]), RuntimeWarning)
            if self.ops[i](current, self.best[i]):
                self.best[i] = current
                flag = True
        if flag:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


def early_stop(monitor=['val_loss', 'val_acc'], patience=10, verbose=VERBOSITY):
    return EarlyStop(monitor=monitor, patience=patience, verbose=verbose)

