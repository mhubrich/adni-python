import random

from cnn.keras import callbacks
from cnn.keras.models.a12.model import build_model
from cnn.keras.optimizers import load_config, MySGD
from cnn.keras.a12.preprocessing.image_processing import inputs
from utils.load_scans import load_scans
from utils.sort_scans import sort_subjects
import sys
sys.stdout = sys.stderr = open('output1', 'w')


# Training specific parameters
target_size = (96, 96)
nb_slices = 1
FRACTION_TRAIN = 0.8
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 64
num_epoch = 500
num_train_samples = 923 * 5  # Per epoch
num_val_samples = 481

# Paths
path_ADNI = '/home/mhubrich/ADNI'
path_checkpoints = '/home/mhubrich/checkpoints/adni/a12_pretrained'
path_weights = None
path_optimizer_weights = None
path_optimizer_updates = None
path_optimizer_config = None


def _split_scans():
    scans = load_scans(path_ADNI)
    subjects, names_tmp = sort_subjects(scans)
    scans_train = []
    scans_val = []
    groups = {}
    for c in classes:
        groups[c] = []
    for n in names_tmp:
        if subjects[n][0].group in classes:
            groups[subjects[n][0].group].append(n)
    min_class = 999999
    for c in groups:
        if len(groups[c]) < min_class:
            min_class = len(groups[c])
    random.seed(SEED)
    for c in groups:
        random.shuffle(groups[c])
        count = 0
        for n in groups[c]:
            for scan in subjects[n]:
                if count < FRACTION_TRAIN * len(groups[c]) and count < FRACTION_TRAIN * min_class:
                    scans_train.append(scan)
                else:
                    scans_val.append(scan)
            count += 1
    return scans_train, scans_val


def train():
    # Get inputs for training and validation
    scans_train, scans_val = _split_scans()
    train_inputs = inputs(scans_train, target_size, nb_slices, batch_size, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, nb_slices, batch_size, classes, 'val', SEED)

    # Set up the model
    model = build_model(num_classes=1, input_shape=(3*nb_slices,) + target_size)
    config = load_config(path_optimizer_config)
    if config == {}:
        config['lr'] = 0.01
        config['decay'] = 0.0001
        config['momentum'] = 0.8
    sgd = MySGD(config, path_optimizer_weights, path_optimizer_updates)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if path_weights:
        model.load_weights(path_weights)

    # Define callbacks
    cbks = [callbacks.checkpoint(path_checkpoints),
            callbacks.save_optimizer(sgd, path_checkpoints, save_only_last=False),
            callbacks.batch_logger(35),
            callbacks.print_history()]

    # Start training
    hist = model.fit_generator(
        train_inputs,
        samples_per_epoch=num_train_samples,
        nb_epoch=num_epoch,
        validation_data=val_inputs,
        nb_val_samples=num_val_samples,
        callbacks=cbks,
        verbose=2,
        max_q_size=320,
        nb_preprocessing_threads=4)

    return hist


if __name__ == "__main__":
    hist = train()

