import keras.backend.common as K
import numpy as np
from keras.preprocessing.image import Iterator
import nibabel as nib
import random

from utils.sort_scans import sort_groups


def _load_scan(scan, voxel, target_size, nb_slices, dim_ordering):
    assert np.min(scan.shape) <= np.min(target_size)
    x_min, x_max = np.min(scan), np.max(scan)
    x_range = (scan.shape[0] - target_size[0]) / 2
    y_range = (scan.shape[1] - target_size[1]) / 2
    assert x_range >= 0 and y_range >= 0
    x_range = range(x_range, scan.shape[0]-x_range)
    y_range = range(y_range, scan.shape[1]-y_range)
    z_range = range(0, scan.shape[2])
    x_slice = range(voxel[0] - (nb_slices / 2), voxel[0] + (nb_slices / 2) + 1)
    y_slice = range(voxel[1] - (nb_slices / 2), voxel[1] + (nb_slices / 2) + 1)
    z_slice = range(voxel[2] - (nb_slices / 2), voxel[2] + (nb_slices / 2) + 1)
    # Get slices
    # xy = scan[x_range, y_range, z_slice]
    xy = scan[:, :, z_slice][:, y_range, :][x_range, :, :]
    # xz = scan[x_range, y_slice, z_range]
    xz = scan[:, y_slice, :][:, :, z_range][x_range, :, :]
    # yz = scan[x_slice, y_range, z_range]
    yz = scan[x_slice, :, :][:, y_range, :][:, :, z_range]
    pad = (target_size[0] - scan.shape[2]) / 2
    xz = np.pad(xz, ((0, 0), (0, 0), (pad, pad)), mode='constant', constant_values=0)
    yz = np.pad(yz, ((0, 0), (0, 0), (pad, pad)), mode='constant', constant_values=0)
    if dim_ordering == 'tf':
        xy = xy.transpose(0, 1, 2)
        xz = xz.transpose(0, 2, 1)
        yz = yz.transpose(1, 2, 0)
        x = np.dstack((xy, xz, yz))
    else:
        xy = xy.transpose(2, 0, 1)
        xz = xz.transpose(1, 0, 2)
        yz = yz.transpose(0, 1, 2)
        x = np.vstack((xy, xz, yz))
    # Transform values in range [0,255]
    x = (x - x_min) / (x_max - x_min)
    x *= 255
    return x


class ScanIterator(Iterator):

    def __init__(self, scans, image_data_generator,
                 target_size=(144, 144), nb_slices=3,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.nb_slices = nb_slices
        self.dim_ordering = dim_ordering
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (3 * nb_slices,)
        else:
            self.image_shape = (3 * nb_slices,) + self.target_size
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
        self.scans = np.zeros((self.nb_sample,) + (160, 160, 96), dtype='float32')
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for c in classes:
            for scan in groups[c]:
                self.classes[i] = self.class_indices[scan.group]
                assert self.classes[i] is not None, \
                    'Read unknown class: %s' % scan.group
                # Load scan and convert to numpy array
                s = nib.load(scan.path).get_data()
                # Remove empty dimension: (160, 160, 96, 1) -> (160, 160, 96)
                s = np.squeeze(s)
                self.scans[i] = s
                self.filenames.append(scan.path)
                i += 1
        random.seed(seed)
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            voxel = (random.randint(23, 72), random.randint(23, 72), random.randint(23, 72))
            x = _load_scan(scan=self.scans[j], voxel=voxel, target_size=self.target_size,
                           nb_slices=self.nb_slices, dim_ordering=self.dim_ordering)
            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
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
