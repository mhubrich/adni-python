from keras.preprocessing.image import ImageDataGenerator
from cnn.keras.metaROI_autoencoder.metaROI2.preprocessing.scan_iterator import ScanIterator


class ScanDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, scans,
                            target_size=(8, 8, 8), classes=None,
                            batch_size=32, shuffle=True, seed=None):
        return ScanIterator(
            scans, self,
            target_size=target_size,
            dim_ordering=self.dim_ordering,
            classes=classes,
            batch_size=batch_size, shuffle=shuffle, seed=seed)
