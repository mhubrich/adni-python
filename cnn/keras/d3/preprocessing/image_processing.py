from cnn.keras.utils.scan_generator_intnorm_position import ScanGenerator
from cnn.keras.utils.filename_generator_intnorm_position import FilenameGenerator


def _image_processing(method):
    if method == 'train':
        generator = ScanGenerator()
    elif method == 'val':
        generator = ScanGenerator()
    else:
        generator = FilenameGenerator()
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
    if method in ['train', 'val']:
        return images.flow_from_directory(
            scans=scans,
            target_size=target_size,
            batch_size=batch_size,
            load_all_scans=load_all_scans,
            classes=classes,
            class_mode='categorical',
            shuffle=shuffle,
            seed=seed)
    else:
        return images.flow_from_directory(scans=scans, grid=seed,
                                          target_size=target_size, load_all_scans=load_all_scans,
                                          classes=classes, batch_size=batch_size)

