import numpy as np
from cnn.keras.AAL.scan_generator import ScanGenerator as Generator


class ScanGenerator(Generator):
    def random_transform(self, x, i=None):
        return np.rot90(x, i)

