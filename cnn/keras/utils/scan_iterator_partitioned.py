import keras.backend as K
import numpy as np

from cnn.keras.utils.scan_iterator import ScanIterator


class ScanIteratorPartitioned(ScanIterator):
    def __init__(self, scans, image_data_generator, partitions,
                 target_size=(30, 30, 30), load_all_scans=True,
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        self.partitions = partitions
        super(ScanIteratorPartitioned, self).__init__(scans=scans, image_data_generator=image_data_generator,
                                                      target_size=target_size, load_all_scans=load_all_scans,
                                                      dim_ordering=dim_ordering,
                                                      classes=classes, class_mode=class_mode,
                                                      batch_size=batch_size, shuffle=shuffle, seed=seed)

    def get_voxel(self, target_size):
        return [self.rand_voxel(target_size, z=Z) for Z in self.partitions]

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = [np.zeros((current_batch_size,) + self.image_shape) for _ in range(len(self.partitions))]
        # build batch of image data
        for i, j in enumerate(index_array):
            for k in range(len(self.partitions)):
                if self.shuffle:
                    voxel = self.rand_voxel(self.target_size, z=self.partitions[k])
                else:
                    voxel = self.voxels[j][k]
                x = self.get_scan(scan=self.scans[j], load_all_scans=self.load_all_scans,
                                  voxel=voxel, target_size=self.target_size)
                if self.shuffle:
                    x = self.image_data_generator.random_transform(x)
                # x = self.image_data_generator.standardize(x)
                x = self.expand_dims(x, self.dim_ordering)
                batch_x[k][i] = x
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
