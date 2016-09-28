from keras.preprocessing.image import ImageDataGenerator

# Paths to train, validation and test folders
TRAIN_DIR = '/home/mhubrich/adni_rotation/'
VALIDATION_DIR = '/home/mhubrich/adni_rotation/'
TEST_DIR = '/home/mhubrich/adni_rotation/'

# For VGG16, images have to be (224,224,3)
IMG_WIDTH = 224
IMG_HEIGHT = 224
COLOR_MODE = 'rgb'


def _image_processing(method):
    if method == 'train':
        # TODO
        # width_shift_range
        # height_shift_range
        # vertical_flip
        # zca_whitening
        generator = ImageDataGenerator()
    elif method == 'val':
        generator = ImageDataGenerator()
    else:
        generator = ImageDataGenerator()
    return generator


def inputs(batch_size, classes, method):
    assert method in ['train', 'val', 'test'], \
        'method must be one of: train, val, test.'

    if method == 'train':
        shuffle = True
        source_dir = TRAIN_DIR
    elif method == 'val':
        shuffle = False
        source_dir = VALIDATION_DIR
    else:
        shuffle = False
        source_dir = TEST_DIR

    image_augmentation = _image_processing(method)
    inputs = image_augmentation.flow_from_directory(
        source_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode=COLOR_MODE,
        batch_size=batch_size,
        classes=classes,
        class_mode='categorical',
        shuffle=shuffle)
    return inputs
