from cnn.keras import callbacks
from cnn.keras.models.vgg16.model import build_model
from cnn.keras.optimizers import load_config, MySGD
from cnn.keras.objectives import mil_mean_squared_error
from cnn.keras.preprocessing.mil.image_processing import inputs


# Training specific parameters
classes = ['Normal', 'AD']
batch_size = 4  # Number of slices for each scan (instances)
num_epoch = 2
num_train_samples = 4  # Number of scans (bags)
num_val_samples = 4
# Paths
path_train = ''
path_val = ''
path_checkpoints = '/tmp/checkpoints'
path_weights = ''
path_optimizer_weights = None
path_optimizer_updates = None
path_optimizer_config = None


def train():
    # Get inputs for training and validation
    train_inputs = inputs(path_train, batch_size, classes, 'train')
    val_inputs = inputs(path_val, batch_size, classes, 'val')

    # Set up the model
    model = build_model(num_classes=len(classes))
    config = load_config(path_optimizer_config)
    sgd = MySGD(config, path_optimizer_weights, path_optimizer_updates)
    model.compile(loss=mil_mean_squared_error, optimizer=sgd, metrics=['accuracy'])
    if path_weights:
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
        nb_preprocessing_threads=4)

    return hist


if __name__ == "__main__":
    hist = train()
