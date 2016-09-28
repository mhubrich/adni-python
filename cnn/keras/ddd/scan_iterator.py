"""
For more information, see:
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L492
"""
from keras.preprocessing.image import Iterator
import keras.backend.common as K
import nibabel as nib
from utils.sort_scans import sort_groups
import numpy as np


def load_scan(filename):
    # Load scan and convert to numpy array
    x = nib.load(filename).get_data()
    # Remove empty dimension: (160, 160, 96, 1) -> (160, 160, 96)
    x = np.squeeze(x)
    return x


class ScanIterator(Iterator):
    def __init__(self, scans,
                 target_size=(160, 160, 96), color_mode='greyscale',
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=16, shuffle=True, seed=None):
        self.target_size = tuple(target_size)
        if color_mode not in {'greyscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "greyscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.dim_ordering not in ['th', 'tf']:
            raise Exception('Unknown dim_ordering: ', self.dim_ordering)
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
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
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for c in classes:
            for scan in groups[c]:
                self.classes[i] = self.class_indices[scan.group]
                assert self.classes[i] is not None, \
                    'Read unknown class: %s' % scan.group
                self.filenames.append(scan.path)
                i += 1
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            x = load_scan(self.filenames[j])
            x_min, x_max = np.min(x), np.max(x)
            x = (x - x_min) / (x_max - x_min)
            x *= 255
            x = np.expand_dims(x, axis=0)
            batch_x[i] = x
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
