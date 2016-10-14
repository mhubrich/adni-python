import os
import pickle


def save_config(path=None):
    config = {'nii': False,
              'ADNI': '/home/mhubrich/ADNI_npy',
              'load_all_scans': False
              }
    if path is None:
        pickle.dump(config, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.p'), 'wb'))
    else:
        pickle.dump(config, open(path, 'wb'))


def load_config(path=None):
    try:
        if path is None:
            return pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.p'), 'rb'))
        else:
            return pickle.load(open(path, 'rb'))
    except IOError:
        return None


config = load_config()