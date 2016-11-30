from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np

from utils.sort_scans import sort_groups
#from cnn.keras.AAL.balanced_class_iterator import BalancedClassIterator


class ScanIterator(Iterator):
    def __init__(self, scans, image_data_generator,
                 target_size=(13, 13, 13), load_all_scans=False,
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
            self.scans1 = np.zeros((self.nb_sample,) + (18, 18, 18), dtype='float32')
            self.scans2 = np.zeros((self.nb_sample,) + (22, 22, 22), dtype='float32')
            self.scans3 = np.zeros((self.nb_sample,) + (33, 33, 33), dtype='float32')
            self.scans4 = np.zeros((self.nb_sample,) + (10, 10, 10), dtype='float32')
            self.scans5 = np.zeros((self.nb_sample,) + (21, 21, 21), dtype='float32')
            self.scans6 = np.zeros((self.nb_sample,) + (15, 15, 15), dtype='float32')
        else:
            self.scans1 = []
            self.scans2 = []
            self.scans3 = []
            self.scans4 = []
            self.scans5 = []
            self.scans6 = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        self.diff = np.zeros((self.nb_sample, 6), dtype=np.float32)
        class_pos = [0]
        i = 0
        for c in classes:
            for scan in groups[c]:
                self.classes[i] = self.class_indices[scan.group]
                assert self.classes[i] is not None, \
                    'Read unknown class: %s' % scan.group
                if self.load_all_scans:
                    self.scans1[i] = self.load_scan(scan.path)
                    self.scans2[i] = self.load_scan(scan.path.replace('meanROI2_1', 'meanROI2_2'))
                    self.scans3[i] = self.load_scan(scan.path.replace('meanROI2_1', 'meanROI2_3'))
                    self.scans4[i] = self.load_scan(scan.path.replace('meanROI2_1', 'meanROI2_4'))
                    self.scans5[i] = self.load_scan(scan.path.replace('meanROI2_1', 'meanROI2_5'))
                    self.scans6[i] = self.load_scan(scan.path.replace('meanROI2_1', 'meanROI2_6'))
                else:
                    self.scans1.append(scan.path)
                    self.scans2.append(scan.path.replace('meanROI2_1', 'meanROI2_2'))
                    self.scans3.append(scan.path.replace('meanROI2_1', 'meanROI2_3'))
                    self.scans4.append(scan.path.replace('meanROI2_1', 'meanROI2_4'))
                    self.scans5.append(scan.path.replace('meanROI2_1', 'meanROI2_5'))
                    self.scans6.append(scan.path.replace('meanROI2_1', 'meanROI2_6'))
                self.diff[i] = np.load(scan.path.replace('meanROI2_1', 'meanROI2_diff'))
                i += 1
            class_pos.append(i)
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
        batch_x1 = np.zeros((current_batch_size,) + (1, 18, 18, 18))
        batch_x2 = np.zeros((current_batch_size,) + (1, 22, 22, 22))
        batch_x3 = np.zeros((current_batch_size,) + (1, 33, 33, 33))
        batch_x4 = np.zeros((current_batch_size,) + (1, 10, 10, 10))
        batch_x5 = np.zeros((current_batch_size,) + (1, 21, 21, 21))
        batch_x6 = np.zeros((current_batch_size,) + (1, 15, 15, 15))
        batch_diff = np.zeros((current_batch_size, 6))
        # build batch of image data
        for i, j in enumerate(index_array):
            x1 = self.get_scan(self.scans1[j])
            x2 = self.get_scan(self.scans2[j])
            x3 = self.get_scan(self.scans3[j])
            x4 = self.get_scan(self.scans4[j])
            x5 = self.get_scan(self.scans5[j])
            x6 = self.get_scan(self.scans6[j])
            x1 = self.expand_dims(x1, self.dim_ordering)
            x2 = self.expand_dims(x2, self.dim_ordering)
            x3 = self.expand_dims(x3, self.dim_ordering)
            x4 = self.expand_dims(x4, self.dim_ordering)
            x5 = self.expand_dims(x5, self.dim_ordering)
            x6 = self.expand_dims(x6, self.dim_ordering)
            batch_x1[i] = x1
            batch_x2[i] = x2
            batch_x3[i] = x3
            batch_x4[i] = x4
            batch_x5[i] = x5
            batch_x6[i] = x6
            batch_diff[i] = self.diff[j]
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
            return [batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_diff]
        return [batch_x1, batch_x2, batch_x3, batch_x4, batch_x5, batch_x6, batch_diff], batch_y

