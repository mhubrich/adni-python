import random

from cnn.keras import callbacks
from cnn.keras.models.metaROI4.auto_model1 import build_model
from cnn.keras.optimizers import load_config, MySGD
from cnn.keras.metaROI_autoencoder.metaROI4.preprocessing.image_processing import inputs
from utils.split_scans import read_imageID

# Training specific parameters
target_size = (13, 13, 13)
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 4
num_epoch = 1500
# Number of training samples per epoch
num_train_samples = 827
# Number of validation samples per epoch
num_val_samples = 451
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_metaROI4'
path_checkpoints = '/home/mhubrich/checkpoints/adni/metaROI4_auto1_1'
path_weights = None
path_optimizer_weights = None
path_optimizer_updates = None
path_optimizer_config = None


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/train_intnorm')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/val_intnorm')
    train_inputs = inputs(scans_train, target_size, batch_size, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, batch_size, classes, 'val', SEED)

    # Set up the model
    model = build_model(input_shape=(1,)+target_size)
    config = load_config(path_optimizer_config)
    if config == {}:
        config['lr'] = 0.1
        config['decay'] = 0.000001
        config['momentum'] = 0.9
    sgd = MySGD(config, path_optimizer_weights, path_optimizer_updates)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    if path_weights:
        model.load_weights(path_weights)

    # Define callbacks
    cbks = [callbacks.checkpoint(path_checkpoints),
            callbacks.save_optimizer(sgd, path_checkpoints, save_only_last=True),
            callbacks.batch_logger(110),
            callbacks.print_history()]

    # Start training
    model.fit_generator(
        train_inputs,
        samples_per_epoch=train_inputs.nb_sample,
        nb_epoch=num_epoch,
        validation_data=val_inputs,
        nb_val_samples=val_inputs.nb_sample,
        callbacks=cbks,
        verbose=2,
        max_q_size=4,
        nb_worker=1,
        pickle_safe=True)


if __name__ == "__main__":
    train()
