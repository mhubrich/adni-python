from cnn.keras.AAL.scan_generator import ScanGenerator


def _image_processing(method):
    if method == 'train':
        generator = ScanGenerator()
    elif method == 'val':
        generator = ScanGenerator()
    else:
        generator = ScanGenerator()
    return generator


def inputs(scans, target_size, batch_size, load_all_scans, classes, method, seed=None, class_mode='categorical'):
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
            class_mode=class_mode,
            shuffle=shuffle,
            seed=seed)
    else:
        return images.flow_from_directory(
            scans=scans,
            target_size=target_size,
            batch_size=batch_size,
            load_all_scans=load_all_scans,
            classes=classes,
            class_mode=None,
            shuffle=False,
            seed=seed)

