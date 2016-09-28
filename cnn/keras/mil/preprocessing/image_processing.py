from cnn.keras.mil.preprocessing.mil_scan_data_generator import MILScanDataGenerator


def _image_processing(method):
    if method == 'train':
        generator = MILScanDataGenerator()
    elif method == 'val':
        generator = MILScanDataGenerator()
    else:
        generator = MILScanDataGenerator()
    return generator


def inputs(scans, target_size, nb_slices, batch_size, classes, method, seed=None):
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
        target_size=target_size,
        nb_slices=nb_slices,
        batch_size=batch_size,
        classes=classes,
        class_mode='binary',
        shuffle=shuffle,
        seed=seed)
    return inputs
