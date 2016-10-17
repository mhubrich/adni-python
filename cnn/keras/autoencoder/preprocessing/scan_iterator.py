import keras.backend as K
import numpy as np

from cnn.keras.utils.scan_iterator import ScanIterator as Iterator


class ScanIterator(Iterator):
    def __init__(self, scans, image_data_generator,
                 target_size=(44, 52, 44), load_all_scans=True,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        super(ScanIterator, self).__init__(scans=scans, image_data_generator=image_data_generator,
                                           target_size=target_size, load_all_scans=load_all_scans,
                                           dim_ordering=dim_ordering,
                                           classes=classes, class_mode=class_mode,
                                           batch_size=batch_size, shuffle=shuffle, seed=seed)

    def get_scan(self, scan, voxel, target_size):
        if not isinstance(scan, np.ndarray):
            scan = self.load_scan(scan)
        return scan[47:, :, :] \
                   [:, 57:, :] \
                   [:, :, 47:]

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            x = self.get_scan(scan=self.scans[j], voxel=None, target_size=None)
            # x = self.image_data_generator.random_transform(x)
            # x = self.image_data_generator.standardize(x)
            x = self.expand_dims(x, self.dim_ordering)
            batch_x[i] = x
        return batch_x, batch_x
