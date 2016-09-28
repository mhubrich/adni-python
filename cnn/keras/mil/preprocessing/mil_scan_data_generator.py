from keras.preprocessing.image import ImageDataGenerator
from cnn.keras.mil.preprocessing.mil_scan_iterator import MILScanIterator


class MILScanDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, scans,
                            target_size=(96, 96), nb_slices=1,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix=None, save_format=None):
        return MILScanIterator(
            scans, self,
            target_size=target_size, nb_slices=nb_slices,
            dim_ordering=self.dim_ordering,
            classes=classes, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed)
