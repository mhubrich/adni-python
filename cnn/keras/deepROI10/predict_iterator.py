from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np


class ScanIterator(Iterator):
    def __init__(self, scan,
                 filter_length, target_size=(22, 22, 22),
                 dim_ordering=K.image_dim_ordering,
                 batch_size=32, seed=None):
        self.target_size = tuple(target_size)
        self.dim_ordering = dim_ordering
        self.indices = np.where(np.ones(self.target_size) > 0)
        self.filter_length = filter_length
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size

        np.random.seed(seed)

        self.scan = self.load_scan(scan.path)
        self.scan_min = np.min(self.scan)
        self.scan_max = np.max(self.scan)
        self.scan_mid = (self.scan_max - self.scan_min) / 2.0
        self.nb_sample = len(self.indices[0]) + 1
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, False, seed)

    def load_scan(self, path):
            return np.load(path)

    def expand_dims(self, x, dim_ordering):
        if dim_ordering == 'tf':
            return np.expand_dims(x, axis=3)
        else:
            return np.expand_dims(x, axis=0)

    # n is assumed to be odd
    def extinguish(self, scan, pos=(0,0,0), n=3):
        x1 = max(0, pos[0] - n/2)
        x2 = min(self.target_size[0], pos[0] + n/2 + 1)
        y1 = max(0, pos[1] - n/2)
        y2 = min(self.target_size[1], pos[1] + n/2 + 1)
        z1 = max(0, pos[2] - n/2)
        z2 = min(self.target_size[2], pos[2] + n/2 + 1)
        if np.mean(scan[x1:x2, y1:y2, z1:z2]) < self.scan_mid:
            scan[x1:x2, y1:y2, z1:z2] = self.scan_max
        else:
            scan[x1:x2, y1:y2, z1:z2] = self.scan_min
        return scan

    def next(self):
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            x = np.array(self.scan, copy=True)
            if j < len(self.indices[0]):
                x = self.extinguish(x, pos=(self.indices[0][j], self.indices[1][j], self.indices[2][j]), n=self.filter_length)
            x = self.expand_dims(x, self.dim_ordering)
            batch_x[i] = x
        return batch_x

