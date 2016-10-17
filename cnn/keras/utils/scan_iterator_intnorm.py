from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np
import nibabel as nib

from utils.sort_scans import sort_groups
from utils.config import config


class ScanIterator(Iterator):
    def __init__(self, scans, image_data_generator,
                 target_size=(30, 30, 30), load_all_scans=True,
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
        self.filenames = []
        if self.load_all_scans:
            self.scans = np.zeros((self.nb_sample,) + (91, 109, 91), dtype='float32')
        else:
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
                if self.load_all_scans:
                    self.scans[i] = self.load_scan(scan.path)
                else:
                    self.scans.append(scan.path)
                self.filenames.append(self.get_filename(scan))
                if not self.shuffle:
                    self.voxels.append(self.get_voxel(self.target_size))
                i += 1
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def load_scan(self, path):
        if config['nii']:
            # Load scan and convert to numpy array
            s = nib.load(path).get_data()
            # Remove empty dimension: (91, 109, 91, 1) -> (91, 109, 91)
            s = np.squeeze(s)
            s_min, s_max = np.min(s), np.max(s)
            # Rescale to [0,1]
            s = (s - s_min) / (s_max - s_min)
            return s
        else:
            return np.load(path)

    def get_filename(self, scan):
        return scan.group + '_' + scan.imageID + '_' + scan.subject

    def get_scan(self, scan, voxel, target_size):
        if not isinstance(scan, np.ndarray):
            scan = self.load_scan(scan)
        return scan[voxel[0]:voxel[0] + target_size[0], :, :] \
                   [:, voxel[1]:voxel[1] + target_size[1], :] \
                   [:, :, voxel[2]:voxel[2] + target_size[2]]

    def expand_dims(self, x, dim_ordering):
        if dim_ordering == 'tf':
            return np.expand_dims(x, axis=3)
        else:
            return np.expand_dims(x, axis=0)

    # old: x=(0, 90), y=(0, 108), z=(0, 90)
    def rand_voxel(self, target_size, x=(5, 82), y=(8, 100), z=(0, 82)):
        """
        Ranges x, y and z are [inclusive, inclusive].
        """
        # randint is [inclusive, exlusive)
        return np.random.randint(x[0], x[1]+2-target_size[0]), np.random.randint(y[0], y[1]+2-target_size[1]),\
               np.random.randint(z[0], z[1]+2-target_size[2])

    def get_voxel(self, target_size):
        return self.rand_voxel(target_size)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            if self.shuffle:
                voxel = self.rand_voxel(self.target_size)
            else:
                voxel = self.voxels[j]
            x = self.get_scan(scan=self.scans[j], voxel=voxel, target_size=self.target_size)
            if self.shuffle:
                x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            x = self.expand_dims(x, self.dim_ordering)
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
