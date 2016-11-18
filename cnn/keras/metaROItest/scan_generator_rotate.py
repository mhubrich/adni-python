import numpy as np
from cnn.keras.metaROItest.scan_generator import ScanGenerator as Generator


class ScanGenerator(Generator):
    def random_transform(self, x, i=None):
        # v1
        #x = np.swapaxes(x, 1, 2)
        #x = np.rot90(x, np.random.randint(4))
        #x = np.swapaxes(x, 1, 2)
        #x = np.swapaxes(x, 0, 2)
        #x = np.rot90(x, np.random.randint(4))
        #x = np.swapaxes(x, 0, 2)
        return np.rot90(x, np.random.randint(4))
        # v2
        #if np.random.randint(6) < 2:
        #    if np.random.randint(2) == 0:
        #        return np.swapaxes(x, 1, 2)
        #    else:
        #        return np.rot90(np.swapaxes(x, 1, 2), 2)
        #else:
        #    return np.rot90(x, np.random.randint(4))
        # v3
                #x = np.rot90(x, np.random.randint(4))
        #x = np.swapaxes(x, 1, 2)
        # v4
        #i = np.random.randint(4)
        #if i == 0:
        #    return np.swapaxes(x, 1, 2)
        #elif i == 1:
        #    return np.rot90(np.swapaxes(x, 1, 2), 2)
        #elif i == 2:
        #    return np.rot90(x, 2)
        #else:
        #    return x
        # v5
        #i = np.random.randint(2)
        #if i == 0:
        #    return np.rot90(x, 2)
        #else:
        #    return x
        # v6
	#if np.random.randint(2) < 1:
        #    return np.rot90(x, 2)
	#else:
	#    return x

