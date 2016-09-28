"""
For more information, see:
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L492
"""
import os

import keras.backend.common as K
import numpy as np
from keras.preprocessing.image import Iterator

import cnn.keras.preprocessing.scans as scans
from utils.sort_scans import sort_groups


class ScanIterator(Iterator):
    def __init__(self, scans, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering=K.image_dim_ordering,
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 slices=range(20, 80), augmentation=False):
        self.image_data_generator = image_data_generator
        self.augmentation = augmentation
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.dim_ordering not in ['th', 'tf']:
            raise Exception('Unknown dim_ordering: ', self.dim_ordering)
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

        # first, count the number of samples and classes
        self.nb_sample = 0
        groups, names = sort_groups(scans)
        if not classes:
            classes = names
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for c in classes:
            assert groups[c] is not None, \
                'Could not find class %s' % c
            assert len(groups[c]) > 0, \
                'Could not find any scans for class %s' % c
            self.nb_sample += len(groups[c])
        print('Found %d scans belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # For each scan, we extract len(slices) slices
        self.nb_sample *= len(slices)
        print('Total amount of images extracted from those scans: %d' % self.nb_sample)

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.slices = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for c in classes:
            for scan in groups[c]:
                for k in slices:
                    self.classes[i] = self.class_indices[scan.group]
                    assert self.classes[i] is not None, \
                        'Read unknown class: %s' % scan.group
                    self.filenames.append(scan.path)
                    self.slices.append(k)
                    i += 1
        super(ScanIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            x = scans.load_scan(self.filenames[j])
            if self.augmentation:
                x = scans.augment_scan(x)
            x = scans.scan_to_img(x, self.target_size, self.slices[j], self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            # Pre-processing of VGG16 trained on ImageNet
            #x[:, :, 0] -= 103.939  # TODO Check order
            #x[:, :, 1] -= 116.779  # TODO Check clipping (min value still 0)
            #x[:, :, 2] -= 123.68
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = scans.array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
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