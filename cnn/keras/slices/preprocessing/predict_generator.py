from keras.preprocessing.image import ImageDataGenerator, Iterator
import keras.backend as K
import numpy as np
import nibabel as nib

from utils.sort_scans import sort_groups
from cnn.keras.slices.preprocessing.scan_iterator import _load_scan, SLICES


target_size = (21, 21, 21)
#interval = range(17, 78 - target_size[0], 5)
#interval = range(13, 81 - target_size[0], 2)
interval = range(17, 78 - target_size[0], 3)
interval_z = [SLICES[0]]
#interval_x = [46, 47, 48, 49, 50]
#interval_y = [44, 45, 46, 47, 48]
#interval_z = [54, 55, 56, 57, 58]
GRID = [(x, y, z) for x in interval for y in interval for z in interval_z]


class PredictGenerator(ImageDataGenerator):
    def flow_from_directory(self, scans,
                            target_size=(30, 30, 30), color_mode='grayscale',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix=None, save_format=None):
        return FilenameIterator(
            scans, self,
            target_size=target_size, batch_size=batch_size,
            dim_ordering=self.dim_ordering,
            classes=classes, class_mode=class_mode)


class FilenameIterator(Iterator):

    def __init__(self, scans, image_data_generator,
                 target_size=(30, 30, 30), batch_size=96,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical'):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
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
        self.scans = np.zeros((self.nb_sample,) + (96, 96, len(SLICES)), dtype='float32')
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        self.nb_sample *= len(GRID)
        print('Extracting %d samples per scan, %d in total.' % (len(GRID), self.nb_sample))
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
                s = s[32:128, :, :][:, 32:128, SLICES]
                # Rescale to [0,1]
                s = (s - s_min) / (s_max - s_min)
                self.scans[i] = s
                self.filenames.append(scan.group + '_' + scan.imageID + '_' + scan.subject)
                i += 1
        super(FilenameIterator, self).__init__(self.nb_sample, batch_size, False, None)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = []
        # build batch of image data
        for i, j in enumerate(index_array):
            j_scan, j_voxel = divmod(j, len(GRID))
            voxel = GRID[j_voxel]
            x = _load_scan(scan=self.scans[j_scan], voxel=voxel, target_size=self.target_size,
                           dim_ordering=self.dim_ordering)
            batch_x[i] = x
            batch_y.append(str(voxel[0]) + '_' + str(voxel[1]) + '_' + str(voxel[2]) + '_' + self.filenames[j_scan])
        return batch_x, batch_y

