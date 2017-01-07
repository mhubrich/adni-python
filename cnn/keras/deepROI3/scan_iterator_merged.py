from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np

from utils.sort_scans import sort_groups


class ScanIterator(Iterator):
    def __init__(self, scans, image_data_generator,
                 target_size=(22, 22, 22), load_all_scans=False,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.load_all_scans = load_all_scans
        self.dim_ordering = dim_ordering
        self.shuffle = shuffle
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        np.random.seed(seed)

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
            self.scansNC = np.zeros((self.nb_sample,) + (17, 17, 17), dtype='float32')
            self.scansAD = np.zeros((self.nb_sample,) + (17, 17, 17), dtype='float32')
            self.scans = np.zeros((self.nb_sample,) + (22, 22, 22), dtype='float32')
        else:
            self.scansNC = []
            self.scansAD = []
            self.scans = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for c in classes:
            for scan in groups[c]:
                self.classes[i] = self.class_indices[scan.group]
                assert self.classes[i] is not None, \
                    'Read unknown class: %s' % scan.group
                if self.load_all_scans:
                    self.scansNC[i] = self.load_scan(scan.path)
                    self.scansAD[i] = self.load_scan(scan.path.replace('deepROI3_NC', 'deepROI3_AD'))
                    self.scans[i] = self.load_scan(scan.path.replace('deepROI3_NC', 'avgpool444'))
                else:
                    self.scansNC.append(scan.path)
                    self.scansAD.append(scan.path.replace('deepROI3_NC', 'deepROI3_AD'))
                    self.scans.append(scan.path.replace('deepROI3_NC', 'avgpool444'))
                i += 1
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

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_xNC = np.zeros((current_batch_size,) + (1, 17, 17, 17))
        batch_xAD = np.zeros((current_batch_size,) + (1, 17, 17, 17))
        batch_x = np.zeros((current_batch_size,) + (1, 22, 22, 22))
        # build batch of image data
        for i, j in enumerate(index_array):
            xNC = self.get_scan(self.scansNC[j])
            xAD = self.get_scan(self.scansAD[j])
            x = self.get_scan(self.scans[j])
            xNC = self.expand_dims(xNC, self.dim_ordering)
            xAD = self.expand_dims(xAD, self.dim_ordering)
            x = self.expand_dims(x, self.dim_ordering)
            batch_xNC[i] = xNC
            batch_xAD[i] = xAD
            batch_x[i] = x
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
            return [batch_xNC, batch_xAD, batch_x]
        return [batch_xNC, batch_xAD, batch_x], batch_y

