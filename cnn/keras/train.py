import random

from cnn.keras import callbacks
from cnn.keras.models.vgg16.model import build_model
from cnn.keras.optimizers import load_config, MySGD
from cnn.keras.preprocessing.image_processing import inputs
from utils.load_scans import load_scans
from utils.sort_scans import sort_groups


# Training specific parameters
FRACTION_TRAIN = 0.8
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 4
num_epoch = 2
# Number of training samples per epoch
num_train_samples = 4
# Number of validation samples per epoch
num_val_samples = 4
# Paths
path_ADNI = '/home/markus/Uni/Masterseminar/ADNI_gz/ADNI'
path_checkpoints = '/tmp/checkpoints'
path_weights = '/home/markus/Uni/Masterseminar/adni/cnn/keras/models/vgg16/vgg16_weights_fc1_classes2.h5'
path_optimizer_weights = None
path_optimizer_updates = None
path_optimizer_config = None


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
    train_inputs = inputs(scans_train, batch_size, classes, 'train')
    val_inputs = inputs(scans_val, batch_size, classes, 'val')

    # Set up the model
    model = build_model(num_classes=len(classes))
    config = load_config(path_optimizer_config)
    sgd = MySGD(config, path_optimizer_weights, path_optimizer_updates)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights(path_weights)

    # Define callbacks
    cbks = [#callbacks.checkpoint(path_checkpoints),
            #callbacks.save_optimizer(sgd, path_checkpoints, save_only_last=True),
            callbacks.batch_logger(1),
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
        nb_worker=2,
        pickle_safe=True)

    return hist


if __name__ == "__main__":
    train()
