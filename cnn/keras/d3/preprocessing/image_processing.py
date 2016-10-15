from cnn.keras.utils.scan_generator_intnorm import ScanGenerator
from cnn.keras.d3.preprocessing.predict_generator import PredictGenerator  # TODO Change


def _image_processing(method):
    if method == 'train':
        generator = ScanGenerator()
    elif method == 'val':
        generator = ScanGenerator()
    else:
        generator = PredictGenerator()
    return generator


def inputs(scans, target_size, batch_size, load_all_scans, classes, method, seed=None):
    assert method in ['train', 'val', 'predict'], \
        'method must be one of: train, val, predict.'

    if method == 'train':
        shuffle = True
    elif method == 'val':
        shuffle = False
    else:
        shuffle = False

    images = _image_processing(method)
    return images.flow_from_directory(
        scans=scans,
        target_size=target_size,
        batch_size=batch_size,
        load_all_scans=load_all_scans,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle,
        seed=seed)
