from cnn.keras.preprocessing.scan_data_generator import ScanDataGenerator

# For VGG16, images have to be (224,224,3)
IMG_WIDTH = 224
IMG_HEIGHT = 224
COLOR_MODE = 'rgb'


def _image_processing(method):
    if method == 'train':
        # TODO
        # width_shift_range
        # height_shift_range
        # vertical_flip
        # zca_whitening
        generator = ScanDataGenerator()
    elif method == 'val':
        generator = ScanDataGenerator()
    else:
        generator = ScanDataGenerator()
    return generator


def inputs(scans, batch_size, classes, method):
    assert method in ['train', 'val', 'test'], \
        'method must be one of: train, val, test.'

    if method == 'train':
        shuffle = True
    elif method == 'val':
        shuffle = False
    else:
        shuffle = False

    image_augmentation = _image_processing(method)
    inputs = image_augmentation.flow_from_directory(
        scans,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle)
    return inputs
