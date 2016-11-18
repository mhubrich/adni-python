from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np

from utils.sort_scans import sort_groups


class PredictIterator(Iterator):
    def __init__(self, scans, load_all_scans=False,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical', batch_size=32):
        self.load_all_scans = load_all_scans
        self.dim_ordering = dim_ordering
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        # first, count the number of samples and classes
        self.nb_sample = 0
        groups, names = sort_groups(scans)
        if not classes:
            classes = names
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for c in classes:
            assert groups[c] is not None, \
                'Could not find class %s' % c
            assert len(groups[c]) > 0, \
                'Could not find any scans for class %s' % c
            self.nb_sample += len(groups[c])
        print('Found %d scans belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        if self.load_all_scans:
            self.scans1 = np.zeros((self.nb_sample,) + (8, 8, 8), dtype='float32')
            self.scans2 = np.zeros((self.nb_sample,) + (6, 6, 6), dtype='float32')
            self.scans3 = np.zeros((self.nb_sample,) + (10, 10, 10), dtype='float32')
            self.scans4 = np.zeros((self.nb_sample,) + (13, 13, 13), dtype='float32')
            self.scans5 = np.zeros((self.nb_sample,) + (5, 5, 5), dtype='float32')
        else:
            self.scans1 = []
            self.scans2 = []
            self.scans3 = []
            self.scans4 = []
            self.scans5 = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for c in classes:
            for scan in groups[c]:
                self.classes[i] = self.class_indices[scan.group]
                assert self.classes[i] is not None, \
                    'Read unknown class: %s' % scan.group
                if self.load_all_scans:
                    self.scans1[i] = self.load_scan(scan.path)
                    self.scans2[i] = self.load_scan(scan.path.replace('metaROI1', 'metaROI2'))
                    self.scans3[i] = self.load_scan(scan.path.replace('metaROI1', 'metaROI3'))
                    self.scans4[i] = self.load_scan(scan.path.replace('metaROI1', 'metaROI4'))
                    self.scans5[i] = self.load_scan(scan.path.replace('metaROI1', 'metaROI5'))
                else:
                    self.scans1.append(scan.path)
                    self.scans2.append(scan.path.replace('metaROI1', 'metaROI2'))
                    self.scans3.append(scan.path.replace('metaROI1', 'metaROI3'))
                    self.scans4.append(scan.path.replace('metaROI1', 'metaROI4'))
                    self.scans5.append(scan.path.replace('metaROI1', 'metaROI5'))
                i += 1
        super(PredictIterator, self).__init__(self.nb_sample, batch_size, False, None)

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

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x1 = np.zeros((current_batch_size,) + (1, 8, 8, 8))
        batch_x2 = np.zeros((current_batch_size,) + (1, 6, 6, 6))
        batch_x3 = np.zeros((current_batch_size,) + (1, 10, 10, 10))
        batch_x4 = np.zeros((current_batch_size,) + (1, 13, 13, 13))
        batch_x5 = np.zeros((current_batch_size,) + (1, 5, 5, 5))
        diff = np.zeros((current_batch_size,) + (5,))
        # build batch of image data
        for i, j in enumerate(index_array):
            x1 = self.get_scan(self.scans1[j])
            x2 = self.get_scan(self.scans2[j])
            x3 = self.get_scan(self.scans3[j])
            x4 = self.get_scan(self.scans4[j])
            x5 = self.get_scan(self.scans5[j])
            diff[i,0] = (0.73041594 - np.mean(x1[np.nonzero(x1)]) + 0.16005278) / (0.41822448 + 0.16005278)
            diff[i,1] = (0.67505538 - np.mean(x2[np.nonzero(x2)]) + 0.13958335) / (0.32063761 + 0.13958335)
            diff[i,2] = (0.75825769 - np.mean(x3[np.nonzero(x3)]) + 0.16636729) / (0.45490789 + 0.16636729)
            diff[i,3] = (0.70375049 - np.mean(x4[np.nonzero(x4)]) + 0.15894139) / (0.41179273 + 0.15894139)
            diff[i,4] = (0.66046154 - np.mean(x5[np.nonzero(x5)]) + 0.15360254) / (0.29526761 + 0.15360254)
            x1 = self.expand_dims(x1, self.dim_ordering)
            x2 = self.expand_dims(x2, self.dim_ordering)
            x3 = self.expand_dims(x3, self.dim_ordering)
            x4 = self.expand_dims(x4, self.dim_ordering)
            x5 = self.expand_dims(x5, self.dim_ordering)
            batch_x1[i] = x1
            batch_x2[i] = x2
            batch_x3[i] = x3
            batch_x4[i] = x4
            batch_x5[i] = x5
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((current_batch_size, self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return [batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, diff]
        return [batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, diff], batch_y

