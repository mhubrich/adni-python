from cnn.keras.d25.preprocessing.scan_data_generator import ScanDataGenerator
from cnn.keras.d25.preprocessing.scan_filename_generator import ScanFilenameGenerator


IMG_WIDTH = 144
IMG_HEIGHT = 144


def _image_processing(method):
    if method == 'train':
        generator = ScanDataGenerator()
    elif method == 'val':
        generator = ScanDataGenerator()
    else:
        generator = ScanFilenameGenerator()
    return generator


def inputs(scans, batch_size, classes, method, seed=None):
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
        scans=scans,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        nb_slices=3,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle,
        seed=seed)
    return inputs
