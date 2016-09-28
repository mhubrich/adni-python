from cnn.keras.preprocessing.mil.mil_image_data_generator import MILImageDataGenerator


# For VGG16, images have to be (160,160,3)
IMG_WIDTH = 160
IMG_HEIGHT = 160
COLOR_MODE = 'rgb'


def _image_processing(method):
    if method == 'train':
        # TODO
        # width_shift_range
        # height_shift_range
        # vertical_flip
        # zca_whitening
        generator = MILImageDataGenerator()
    elif method == 'val':
        generator = MILImageDataGenerator()
    else:
        generator = MILImageDataGenerator()
    return generator


def inputs(directory, batch_size, classes, method):
    assert method in ['train', 'val', 'test'], \
        'method must be one of: train, val, test.'

    if method == 'train':
        shuffle = True
    elif method == 'val':
        shuffle = False
    else:
        shuffle = False

    images = _image_processing(method)
    inputs = images.flow_from_directory(
        directory,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle)
    return inputs
