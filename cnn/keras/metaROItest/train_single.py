from keras.optimizers import SGD
from cnn.keras import callbacks
from cnn.keras.models.metaROI2.model_single5 import build_model
from cnn.keras.optimizers import load_config, MySGD
from cnn.keras.metaROItest.image_processing import inputs
from utils.split_scans import read_imageID


# Training specific parameters
target_size = (6, 6, 6)
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = False
num_epoch = 2000
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_metaROI2'
path_checkpoints = '/home/mhubrich/checkpoints/adni/metaROI2_CV3f'
path_weights = None
path_optimizer_weights = None
path_optimizer_updates = None
path_optimizer_config = None


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/CV2/3_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/CV2/3_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'val', SEED)

    # Set up the model
    model = build_model()
    config = load_config(path_optimizer_config)
    if config == {}:
        config['lr'] = 0.001
        config['decay'] = 0.000001
        config['momentum'] = 0.9
    #sgd = MySGD(config, path_optimizer_weights, path_optimizer_updates)
    sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if path_weights:
        model.load_weights(path_weights)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.save_model(path_checkpoints)]
            #callbacks.save_optimizer(sgd, path_checkpoints, save_only_last=True)]
            #callbacks.batch_logger(70),
            #[callbacks.print_history()]

    # Start training
    model.fit_generator(
        train_inputs,
        samples_per_epoch=train_inputs.nb_sample,
        nb_epoch=num_epoch,
        validation_data=val_inputs,
        nb_val_samples=val_inputs.nb_sample,
        callbacks=cbks,
        verbose=2,
        max_q_size=32,
        nb_worker=1,
        pickle_safe=True)


if __name__ == "__main__":
    train()
