from keras.preprocessing.image import ImageDataGenerator
from cnn.keras.slices_merged.preprocessing.scan_iterator import ScanIterator


class ScanDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, scans, slices,
                            target_size=(10, 10, 10), color_mode='grayscale',
                            classes=None, class_mode='categorical',
                            batch_size=64, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix=None, save_format=None):
        return ScanIterator(
            scans, self, slices,
            target_size=target_size,
            dim_ordering=self.dim_ordering,
            classes=classes, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed)
