from cnn.keras.d25.preprocessing.scan_data_generator import ScanDataGenerator
from cnn.keras.d25.preprocessing.scan_iterator import ScanIterator, _load_scan
import keras.backend as K
import numpy as np
import random


class ScanFilenameGenerator(ScanDataGenerator):
    def flow_from_directory(self, scans,
                            target_size=(144, 144), nb_slices=3,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix=None, save_format=None):
        return ScanFilenameIterator(
            scans, self,
            target_size=target_size, nb_slices=nb_slices,
            dim_ordering=self.dim_ordering,
            classes=classes, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed)


class ScanFilenameIterator(ScanIterator):
    def __init__(self, scans, image_data_generator,
                 target_size=(144, 144), nb_slices=3,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        super(ScanFilenameIterator, self).__init__(scans=scans,
                                                   image_data_generator=image_data_generator,
                                                   target_size=target_size, nb_slices=nb_slices,
                                                   dim_ordering=dim_ordering,
                                                   classes=classes, class_mode=class_mode,
                                                   batch_size=batch_size, shuffle=shuffle, seed=seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = []
        # build batch of image data
        for i, j in enumerate(index_array):
            voxel = (random.randint(23, 72), random.randint(23, 72), random.randint(23, 72))
            x = _load_scan(scan=self.scans[j], voxel=voxel, target_size=self.target_size,
                           nb_slices=self.nb_slices, dim_ordering=self.dim_ordering)
            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y.append(self.filenames[j])
        return batch_x, batch_y
