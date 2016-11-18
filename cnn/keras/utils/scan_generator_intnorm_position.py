from keras.preprocessing.image import ImageDataGenerator
from cnn.keras.utils.scan_iterator_intnorm_position import ScanIterator


class ScanGenerator(ImageDataGenerator):
    def flow_from_directory(self, scans,
                            target_size=(30, 30, 30), load_all_scans=True,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix=None, save_format=None):
        return ScanIterator(
            scans, self,
            target_size=target_size, load_all_scans=load_all_scans,
            dim_ordering=self.dim_ordering,
            classes=classes, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed)

    def random_transform(self, x):
        return x
