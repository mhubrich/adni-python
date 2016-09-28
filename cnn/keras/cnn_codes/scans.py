import numpy as np
import nibabel as nib
from PIL import Image


def load_scan(filename):
    # Load scan and convert to numpy array
    x = nib.load(filename).get_data()
    # Remove empty dimension: (160, 160, 96, 1) -> (160, 160, 96)
    x = np.squeeze(x)
    return x


def scan_to_array(scan, nb_slice, dim_ordering):
    # TODO Check intensity normalization
    x_min, x_max = np.min(scan), np.max(scan)
    # Get single slice: (160, 160, 1)
    x = scan[:, :, nb_slice]
    # Transform values in range [0,255]
    x = (x - x_min) / (x_max - x_min)
    x *= 255
    # Convert array to RGB, i.e. all three channels are the same
    x = np.dstack([x] * 3)
    #x = Image.fromarray(x.astype('uint8'), 'RGB').resize(target_size, Image.ANTIALIAS)
    #x.save('/home/markus/test2.jpg', 'JPEG', quality=100)
    #x = np.asarray(x, dtype='float32')
    if dim_ordering == 'th':
        x = x.transpose(2, 0, 1)
    else:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    return x
