from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np
import nibabel as nib
import random
from scipy import ndimage

from utils.sort_scans import sort_groups


def _rand_voxel(target_size):
    return random.randint(7, 86-target_size[0]), random.randint(3, 92-target_size[1]),\
           random.randint(15, 80-target_size[2])
    #return random.randint(0, 96-target_size[0]), random.randint(0, 96-target_size[1]),\
    #       random.randint(0, 96-target_size[2])


def _load_scan(scan, voxel, target_size, dim_ordering):
    x = scan[voxel[0]:voxel[0]+target_size[0], :, :] \
            [:, voxel[1]:voxel[1]+target_size[1], :] \
            [:, :, voxel[2]:voxel[2]+target_size[2]]
    if dim_ordering == 'tf':
        x = np.expand_dims(x, axis=3)
    else:
        x = np.expand_dims(x, axis=0)
    return x


def rotate_scan(scan, dim_ordering):
    scan = np.squeeze(scan)
    scan = np.rot90(scan, random.randint(0, 3))
    if dim_ordering == 'tf':
        scan = np.expand_dims(scan, axis=3)
    else:
        scan = np.expand_dims(scan, axis=0)
    return scan


def augment_scan(scan):  # new
    x = random.random() * 360
    y = random.random() * 360
    z = random.random() * 360
    scan = ndimage.rotate(scan, x, axes=(1, 2), reshape=False, mode='constant', cval=0.0)
    scan = ndimage.rotate(scan, y, axes=(0, 2), reshape=False, mode='constant', cval=0.0)
    scan = ndimage.rotate(scan, z, axes=(0, 1), reshape=False, mode='constant', cval=0.0)
    return scan


class ScanIterator(Iterator):

    def __init__(self, scans, image_data_generator,
                 target_size=(30, 30, 30),
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.shuffle = shuffle
        self.dim_ordering = dim_ordering
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

        random.seed(seed)
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
        self.filenames = []
        self.scans = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        if not self.shuffle:
            self.voxels = []
        i = 0
        for c in classes:
            for scan in groups[c]:
                self.classes[i] = self.class_indices[scan.group]
                assert self.classes[i] is not None, \
                    'Read unknown class: %s' % scan.group
                self.scans.append(scan.path)
                self.filenames.append(scan.group + '_' + scan.imageID + '_' + scan.subject)
                if not self.shuffle:
                    self.voxels.append(_rand_voxel(self.target_size))
                i += 1
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            if self.shuffle:
                voxel = _rand_voxel(self.target_size)
            else:
                voxel = self.voxels[j]
            x = _load_scan(path=self.scans[j], voxel=voxel, target_size=self.target_size,
                           dim_ordering=self.dim_ordering)
            #if self.shuffle:  # new
            #    x = rotate_scan(x, self.dim_ordering)  # new
            #if self.dim_ordering == 'tf':  # new
            #    x = np.expand_dims(x, axis=3)  # new
            #else:  # new
            #    x = np.expand_dims(x, axis=0)  # new
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
