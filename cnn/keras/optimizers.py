from keras import backend as K
from keras.optimizers import RMSprop, SGD
import pickle


def _restore_state(optimizer):
    """
    It is important to first call the super function of 'get_updates'
    and only then restore the state.
    """
    if optimizer.path_weights:
        weights = pickle.load(open(optimizer.path_weights, 'rb'))
        optimizer.set_weights(weights)
    if optimizer.path_updates:
        updates = pickle.load(open(optimizer.path_updates, 'rb'))
        optimizer.set_state(updates)


def load_config(path):
    if path is None:
        return {}
    else:
        return pickle.load(open(path, 'rb'))


def _save_weights(optimizer, path):
    weights = optimizer.get_weights()
    pickle.dump(weights, open(path, 'wb'))


def _save_updates(optimizer, path):
    updates = optimizer.get_state()
    pickle.dump(updates, open(path, 'wb'))


def _save_config(config, path):
    pickle.dump(config, open(path, 'wb'))


class MyRMSprop(RMSprop):
    def __init__(self, config, path_weights=None, path_updates=None, **kwargs):
        self.path_weights = path_weights
        self.path_updates = path_updates
        # Set default values
        if 'lr' not in config:
            config['lr'] = 1e-2
        if 'rho' not in config:
            config['rho'] = 0.9
        if 'epsilon' not in config:
            config['epsilon'] = 1e-8
        super(MyRMSprop, self).__init__(config['lr'], config['rho'], config['epsilon'], **kwargs)

    def get_updates(self, params, constraints, loss):
        tmp = super(MyRMSprop, self).get_updates(params, constraints, loss)
        _restore_state(self)
        return tmp

    def save_weights(self, path):
        _save_weights(self, path)

    def save_updates(self, path):
        _save_updates(self, path)

    def save_config(self, path):
        # TODO Check config
        _save_config(self.get_config(), path)


class MySGD(SGD):
    def __init__(self, config, path_weights=None, path_updates=None, **kwargs):
        self.path_weights = path_weights
        self.path_updates = path_updates
        # Set default values
        if 'lr' not in config:
            config['lr'] = 1e-2
        if 'momentum' not in config:
            config['momentum'] = 0.9
        if 'decay' not in config:
            config['decay'] = 1e-6
        if 'nesterov' not in config:
            config['nesterov'] = True
        super(MySGD, self).__init__(config['lr'], config['momentum'], config['decay'], config['nesterov'], **kwargs)

    def get_updates(self, params, constraints, loss):
        tmp = super(MySGD, self).get_updates(params, constraints, loss)
        _restore_state(self)
        return tmp

    def save_weights(self, path):
        _save_weights(self, path)

    def save_updates(self, path):
        _save_updates(self, path)

    def save_config(self, path):
        config = self.get_config()
        config['lr'] = K.get_value(self.lr) * (1. / (1. + K.get_value(self.decay) * K.get_value(self.iterations)))
        _save_config(config, path)
