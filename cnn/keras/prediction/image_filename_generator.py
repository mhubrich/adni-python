from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator, array_to_img, img_to_array, load_img
import keras.backend.common as K
import numpy as np
import os


class ImageFilenameDataGenerator(ImageDataGenerator):
    def __init__(self):
        super(ImageFilenameDataGenerator, self).__init__()

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return FilenameDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class FilenameDirectoryIterator(DirectoryIterator):
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        super(FilenameDirectoryIterator, self).__init__(directory, image_data_generator,
                                                        target_size=target_size, color_mode=color_mode,
                                                        dim_ordering=dim_ordering,
                                                        classes=classes, class_mode=class_mode,
                                                        batch_size=batch_size, shuffle=shuffle, seed=seed,
                                                        save_to_dir=save_to_dir, save_prefix=save_prefix,
                                                        save_format=save_format)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        batch_y = []
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname), grayscale=grayscale, target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y.append(fname)
        return batch_x, batch_y
