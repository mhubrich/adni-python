from keras.preprocessing.image import ImageDataGenerator
from cnn.keras.utils.filename_iterator_intnorm_pos_diff_mean import FilenameIterator


class FilenameGenerator(ImageDataGenerator):
    def flow_from_directory(self, scans, grid,
                            target_size=(30, 30, 30), load_all_scans=True,
                            classes=None, batch_size=32):
        return FilenameIterator(
            scans, self, grid=grid,
            target_size=target_size, load_all_scans=load_all_scans,
            dim_ordering=self.dim_ordering,
            classes=classes, batch_size=batch_size)
