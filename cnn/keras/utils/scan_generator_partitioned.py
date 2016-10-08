from cnn.keras.utils.scan_generator_rotate import ScanGeneratorRotate
from cnn.keras.utils.scan_iterator_partitioned import ScanIteratorPartitioned


class ScanGeneratorPartitioned(ScanGeneratorRotate):
    def flow_from_directory(self, scans, partitions,
                            target_size=(30, 30, 30), load_all_scans=True,
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix=None, save_format=None):
        return ScanIteratorPartitioned(
            scans, self, partitions=partitions,
            target_size=target_size, load_all_scans=load_all_scans,
            dim_ordering=self.dim_ordering,
            classes=classes, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed)
