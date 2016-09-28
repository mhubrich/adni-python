import random

from cnn.keras.vgg16.ddd.model import build_model

from cnn.keras import callbacks
from cnn.keras.ddd.scan_iterator import ScanIterator
from utils.load_scans import load_scans
from utils.sort_scans import sort_groups

# Training specific parameters
FRACTION_TRAIN = 0.8
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 16
num_epoch = 5
# Number of training samples per epoch
num_train_samples = 200
# Number of validation samples per epoch
num_val_samples = 200
# Paths
path_ADNI = '/home/markus/Uni/Masterseminar/ADNI_gz/ADNI'
path_checkpoints = '/tmp/checkpoints'
path_weights = None
path_optimizer_weights = None
path_optimizer_updates = None


def _split_scans():
    scans = load_scans(path_ADNI)
    groups, _ = sort_groups(scans)
    scans_train = []
    scans_val = []
    for c in classes:
        random.seed(SEED)
        random.shuffle(groups[c])
        count = 0
        for scan in groups[c]:
            if count < len(groups[c]) * FRACTION_TRAIN:
                scans_train.append(scan)
            else:
                scans_val.append(scan)
            count += 1
    return scans_train, scans_val


def train():
    # Get inputs for training and validation
    scans_train, scans_val = _split_scans()
    train_inputs = ScanIterator(scans_train, batch_size=16, shuffle=True, classes=classes)
    val_inputs = ScanIterator(scans_val, batch_size=16, shuffle=False, classes=classes)

    # Set up the model
    model = build_model(num_classes=len(classes))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Define callbacks
    cbks = [callbacks.batch_logger(50),
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
        max_q_size=32,
        nb_preprocessing_threads=4)

    return hist


if __name__ == "__main__":
    hist = train()
