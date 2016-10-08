from cnn.keras.utils.scan_generator import ScanGenerator
import numpy as np


class ScanGeneratorRotate(ScanGenerator):
    def random_transform(self, x):
        return np.rot90(x, np.random.randint(4))
