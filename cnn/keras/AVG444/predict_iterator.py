from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np


class ScanIterator(Iterator):
    def __init__(self, scan, image_data_generator,
                 target_size=(22, 22, 22), load_all_scans=False,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode=None,
                 batch_size=32, shuffle=False, seed=None,
                 filter_length=4):
        self.target_size = tuple(target_size)
        self.dim_ordering = dim_ordering
        self.shuffle = shuffle
        self.filter_length = filter_length
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        np.random.seed(seed)

        self.scans = self.load_scan(scan.path)
        self.classes = 0 if scan.group == 'Normal' else 1
        self.nb_sample = (self.target_size[0] - self.filter_length + 1) ** 3
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def load_scan(self, path):
            return np.load(path)

    def get_scan(self, scan):
        if not isinstance(scan, np.ndarray):
            return self.load_scan(scan)
        return scan

    def expand_dims(self, x, dim_ordering):
        if dim_ordering == 'tf':
            return np.expand_dims(x, axis=3)
        else:
            return np.expand_dims(x, axis=0)

    def mod3(self, a, b):
        c, x = divmod(a, b)
        z, y = divmod(c, b)
        return x, y, z

    def extinguish(self, scan, pos=(0,0,0), n=self.filter_length):
        return scan[pos[0]:pos[0]+n, pos[1]:pos[1]+n, pos[2]:pos[2]+n] = 0

    def next(self):
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + (1, 22, 22, 22))
        # build batch of image data
        for i, j in enumerate(index_array):
            x = np.array(self.scan, copy=True)
            x = self.extinguish(x, pos=mod3(j), n=self.filter_length)
            x = self.expand_dims(x, self.dim_ordering)
            batch_x[i] = x
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = np.tile(self.classes, current_batch_size)
        elif self.class_mode == 'binary':
            batch_y = np.tile(self.classes, current_batch_size).astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((current_batch_size, 2), dtype='float32')
            for i, label in enumerate(np.tile(self.classes, current_batch_size)):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

