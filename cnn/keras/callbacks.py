"""
For further information, see: http://keras.io/callbacks/
"""
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
import os
import time
import sys

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


def early_stopping(num_epochs):
    return EarlyStopping(
        monitor='val_acc',
        patience=num_epochs,
        verbose=VERBOSITY)


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
