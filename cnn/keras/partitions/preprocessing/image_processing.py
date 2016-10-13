from cnn.keras.utils.scan_generator_partitioned import ScanGeneratorPartitioned
from cnn.keras.utils.filename_generator_partitioned import FilenameGeneratorPartitioned


def _image_processing(method):
    if method == 'train':
        generator = ScanGeneratorPartitioned()
    elif method == 'val':
        generator = ScanGeneratorPartitioned()
    else:
        generator = FilenameGeneratorPartitioned()
    return generator


def inputs(scans, target_size, partitions, batch_size, load_all_scans, classes, method, seed=None):
    assert method in ['train', 'val', 'predict'], \
        'method must be one of: train, val, predict.'

    if method == 'train':
        shuffle = True
    else:
        shuffle = False

    images = _image_processing(method)
    if method in ['train', 'val']:
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
    else:
        return images.flow_from_directory(scans=scans, grid=partitions,
                                          target_size=target_size, load_all_scans=load_all_scans,
                                          classes=classes, batch_size=batch_size)
