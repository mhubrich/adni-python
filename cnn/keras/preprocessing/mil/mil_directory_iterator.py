"""
For more information, see:
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L492
"""
import keras.backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img
import numpy as np
import os
import random


class MILDirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(160, 160), color_mode='rgb',
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=16, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.sample_batch_size = batch_size
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        if seed is not None:
            random.seed(seed)

        # first, count the number of classes
        if not classes:
            classes = []
            for subdir in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # second, count the number of samples and bags
        self.nb_sample = 0
        self.nb_bags = 0

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for bag in os.listdir(subpath):
                self.nb_bags += 1
                subsubpath = os.path.join(subpath, bag)
                for fname in os.listdir(subsubpath):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.nb_sample += 1
        print('Found %d images belonging to %d classes in %d bags.' % (self.nb_sample, self.nb_class, self.nb_bags))

        # third, build an index of the images in the different class and bag subfolders
        self.filenames = []
        nb_slices = 40
        self.classes = np.zeros((self.nb_bags,nb_slices), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for bag in os.listdir(subpath):
                self.filenames.append([])
                subsubpath = os.path.join(subpath, bag)
                j = 0
                for fname in os.listdir(subsubpath):
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.filenames[i].append(os.path.join(subdir, bag, fname))
                        self.classes[i][j] = self.class_indices[subdir]
                        j += 1
                i += 1
        super(MILDirectoryIterator, self).__init__(self.nb_bags, 1, shuffle, seed)

    def next(self):
        # At the moment, we restrict len(index_array) = 1
        with self.lock:
            index_array, _, _ = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((self.sample_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        index = index_array[0]
        samples = random.sample(range(len(self.filenames[index])), self.sample_batch_size)
        for i, j in enumerate(samples):
            fname = self.filenames[index][j]
            img = load_img(os.path.join(self.directory, fname), grayscale=grayscale, target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(self.sample_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                          hash=np.random.randint(1e4),
                                                          format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index][samples]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index][samples].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index][samples]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y
