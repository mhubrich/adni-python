from cnn.keras.slices_merged.preprocessing.scan_data_generator import ScanDataGenerator
# from cnn.keras.slices_merged.preprocessing.predict_generator import PredictGenerator


def _image_processing(method):
    if method == 'train':
        generator = ScanDataGenerator()
    elif method == 'val':
        generator = ScanDataGenerator()
    else:
        generator = ScanDataGenerator()
    return generator


def inputs(scans, target_size, slices, batch_size, classes, method, seed=None):
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
        slices=slices,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle,
        seed=seed)
    return inputs
