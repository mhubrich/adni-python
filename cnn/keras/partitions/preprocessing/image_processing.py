from cnn.keras.utils.scan_generator_partitioned import ScanGeneratorPartitioned


def _image_processing(method):
    if method == 'train':
        generator = ScanGeneratorPartitioned()
    elif method == 'val':
        generator = ScanGeneratorPartitioned()
    else:
        generator = ScanGeneratorPartitioned()
    return generator


def inputs(scans, target_size, partitions, batch_size, load_all_scans, classes, method, seed=None):
    assert method in ['train', 'val', 'test'], \
        'method must be one of: train, val, test.'

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
        partitions=partitions,
        batch_size=batch_size,
        load_all_scans=load_all_scans,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle,
        seed=seed)
