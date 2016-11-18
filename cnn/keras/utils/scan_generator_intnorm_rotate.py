import numpy as np
from cnn.keras.utils.scan_generator_intnorm import ScanGenerator as Generator


class ScanGenerator(Generator):
    def random_transform(self, x):
        return np.rot90(x, np.random.randint(4))
