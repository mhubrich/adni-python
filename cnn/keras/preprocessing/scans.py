import numpy as np
import nibabel as nib
from scipy import ndimage
import random


def load_scan(filename):
    # Load scan and convert to numpy array
    x = nib.load(filename).get_data()
    # Remove empty dimension: (160, 160, 96, 1) -> (160, 160, 96)
    x = np.squeeze(x)
    return x


def scan_to_img(scan, target_size, nb_slice, dim_ordering):
    # TODO Check intensity normalization
    x_min, x_max = np.min(scan), np.max(scan)
    # Get single slice: (160, 160, 1)
    x = scan[:, :, nb_slice]
    # Transform values in range [0,255]
    x = (x - x_min) / (x_max - x_min)
    x *= 255
    # Pad the array to get CNN input size, e.g. (160, 160, 1) -> (224, 224, 1)
    assert x.shape[0] <= target_size[0] and x.shape[1] <= target_size[1], \
        'Scan bigger than target size: (%d,%d) vs. (%d,%d).' % (x.shape[0], x.shape[1], target_size[0], target_size[1])
    pad11 = pad12 = (target_size[0] - x.shape[0]) / 2
    pad21 = pad22 = (target_size[1] - x.shape[1]) / 2
    if (target_size[0] - x.shape[0]) % 2 != 0:
        pad12 += 1
    if (target_size[1] - x.shape[1]) % 2 != 0:
        pad22 += 1
    # TODO Test edge and constant
    x = np.pad(x, ((pad11, pad12), (pad21, pad22)), mode='edge')  # , constant_values=0)
    # Convert array to RGB, i.e. all three channels are the same
    x = np.dstack([x] * 3)
    if dim_ordering == 'th':
        x = x.transpose(2, 0, 1)
    else:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    return x


def augment_scan(scan):
    x = random.random() * 360
    y = random.random() * 360
    z = random.random() * 360
    scan = ndimage.rotate(scan, x, axes=(1, 2), reshape=False, mode='constant', cval=0.0)
    scan = ndimage.rotate(scan, y, axes=(0, 2), reshape=False, mode='constant', cval=0.0)
    scan = ndimage.rotate(scan, z, axes=(0, 1), reshape=False, mode='constant', cval=0.0)
    return scan


def array_to_img(x, dim_ordering, scale=True):
    """ Source:
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L121
    """
    from PIL import Image
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    else:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        return Image.fromarray(x.astype('uint8'), 'RGB')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])
