from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np
import nibabel as nib
import random

from utils.sort_scans import sort_groups


def _rand_voxel():
    return random.randint(7, 86), random.randint(3, 92), random.randint(15, 80)


def _load_scan(scan, voxel, nb_slices, dim_ordering):
    x_slice = range(voxel[0] - (nb_slices / 2), voxel[0] + (nb_slices / 2) + 1)
    y_slice = range(voxel[1] - (nb_slices / 2), voxel[1] + (nb_slices / 2) + 1)
    z_slice = range(voxel[2] - (nb_slices / 2), voxel[2] + (nb_slices / 2) + 1)
    # Get slices
    xy = scan[:, :, z_slice]
    xz = scan[:, y_slice, :]
    yz = scan[x_slice, :, :]
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
    return x


class MILScanIterator(Iterator):

    def __init__(self, scans, image_data_generator,
                 target_size=(96, 96), nb_slices=1,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.nb_slices = nb_slices
        self.shuffle = shuffle
        self.instances = batch_size
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

        random.seed(seed)

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
        self.scans = np.zeros((self.nb_sample,) + (96, 96, 96), dtype='float32')
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
                s_min, s_max = np.min(s), np.max(s)
                # Cut slice (160, 160, 96) -> (96, 96, 96)
                s = s[32:128, :, :][:, 32:128, :]
                # Rescale to [0,1]
                s = (s - s_min) / (s_max - s_min)
                self.scans[i] = s
                self.filenames.append(scan.group + '_' + scan.imageID + '_' + scan.subject)
                i += 1
        super(MILScanIterator, self).__init__(self.nb_sample, 1, shuffle, seed)

    def next(self):
        # At the moment, we restrict len(index_array) = 1
        with self.lock:
            index_array, _, _ = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((self.instances,) + self.image_shape)
        scan = index_array[0]
        # build batch of image data
        for i in range(self.instances):
            voxel = _rand_voxel()
            x = _load_scan(scan=self.scans[scan], voxel=voxel, nb_slices=self.nb_slices, dim_ordering=self.dim_ordering)
            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[np.tile(index_array, self.instances)]
        elif self.class_mode == 'binary':
            batch_y = self.classes[np.tile(index_array, self.instances)].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[np.tile(index_array, self.instances)]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
