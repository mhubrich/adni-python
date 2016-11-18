from keras.preprocessing.image import Iterator
import keras.backend as K
import numpy as np
import nibabel as nib

from utils.sort_scans import sort_groups
from utils.config import config


class FilenameIterator(Iterator):
    def __init__(self, scans, image_data_generator, grid,
                 target_size=(30, 30, 30), load_all_scans=True,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, batch_size=32):
        self.image_data_generator = image_data_generator
        self.grid = grid
        self.target_size = tuple(target_size)
        self.load_all_scans = load_all_scans
        self.dim_ordering = dim_ordering
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size

        # first, count the number of samples and classes
        self.nb_sample = 0
        groups, names = sort_groups(scans)
        if not classes:
            classes = names

        for c in classes:
            assert groups[c] is not None, \
                'Could not find class %s' % c
            assert len(groups[c]) > 0, \
                'Could not find any scans for class %s' % c
            self.nb_sample += len(groups[c])
        print('Found %d scans belonging to %d classes.' % (self.nb_sample, len(classes)))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        if self.load_all_scans:
            self.scans = np.zeros((self.nb_sample,) + (88, 108, 88), dtype='float32')
        else:
            self.scans = []
        self.nb_sample *= self.len_grid()
        i = 0
        for c in classes:
            for scan in groups[c]:
                if self.load_all_scans:
                    self.scans[i] = self.load_scan(scan.path)
                else:
                    self.scans.append(scan.path)
                self.filenames.append(self.get_filename(scan))
                i += 1
        super(FilenameIterator, self).__init__(self.nb_sample, batch_size, False, None)

    def load_scan(self, path):
        if config['nii']:
            # Load scan and convert to numpy array
            s = nib.load(path).get_data()
            # Remove empty dimension: (160, 160, 96, 1) -> (160, 160, 96)
            s = np.squeeze(s)
            s_min, s_max = np.min(s), np.max(s)
            # Rescale to [0,1]
            s = (s - s_min) / (s_max - s_min)
            return s
        else:
            return np.load(path)

    def get_filename(self, scan):
        return scan.group + '_' + scan.imageID + '_' + scan.subject

    def get_voxel_str(self, voxel):
        return str(voxel[0]) + '_' + str(voxel[1]) + '_' + str(voxel[2])

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

    def len_grid(self):
        return len(self.grid)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        positions = np.zeros((current_batch_size,) + (3,))
        batch_y = []
        # build batch of image data
        for i, j in enumerate(index_array):
            j_scan, j_voxel = divmod(j, self.len_grid())
            voxel = self.grid[j_voxel]
            x = self.get_scan(scan=self.scans[j_scan], voxel=voxel, target_size=self.target_size)
            # x = self.image_data_generator.standardize(x)
            x = self.expand_dims(x, self.dim_ordering)
            batch_x[i] = x
            positions[i] = [(voxel[0]-5)/(82.0+1-self.target_size[0]-5.0),
                            (voxel[1]-5)/(102.0+1-self.target_size[1]-5.0),
                            (voxel[2]-5)/(82.0+1-self.target_size[2]-5.0)]
            batch_y.append(self.get_voxel_str(voxel) + '_' + self.filenames[j_scan])
        return [batch_x, positions], batch_y
