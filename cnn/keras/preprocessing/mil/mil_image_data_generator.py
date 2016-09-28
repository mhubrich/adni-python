from keras.preprocessing.image import ImageDataGenerator
from cnn.keras.preprocessing.mil.mil_directory_iterator import MILDirectoryIterator


class MILImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory,
                            target_size=(160, 160), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=16, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return MILDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)
