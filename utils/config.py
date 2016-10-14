import pickle


def save_config(path='config.p'):
    config = {'nii': True,
              'ADNI': '/home/mhubrich/ADNI',
              'load_all_scans': True
              }
    pickle.dump(config, open(path, 'wb'))


def load_config(path='config.p'):
    try:
        return pickle.load(open(path, 'rb'))
    except IOError:
        return None


config = load_config()