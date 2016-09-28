from cnn.keras import callbacks
from cnn.keras.models.vgg16.model import build_model
from cnn.keras.optimizers import MyRMSprop
from cnn.keras.rotation import inputs

# Training specific parameters
classes = ['Normal', 'AD']
batch_size = 16
num_epoch = 10
# Number of training samples per epoch
num_train_samples = 20000
# Number of validation samples per epoch
num_val_samples = 4000
# Paths
path_checkpoints = '/tmp/checkpoints'
path_weights = 'vgg16_weights_fc1_classes2.h5'
path_optimizer_weights = None
path_optimizer_updates = None


def train(weights_path):
    # Get inputs for training and validation
    train_inputs = inputs(batch_size, classes, 'train')
    val_inputs = inputs(batch_size, classes, 'val')

    # Set up the model
    model = build_model(2)
    rms = MyRMSprop(path_optimizer_weights, path_optimizer_updates)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    model.load_weights(path_weights)

    # Define callbacks
    cbks = [callbacks.checkpoint(path_checkpoints),
            callbacks.save_optimizer(rms, path_checkpoints, save_only_last=True),
            callbacks.learning_rate(lr=0.01, decay_rate=0.1, decay_epochs=10),
            callbacks.batch_logger(100),
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
        max_q_size=256,
        nb_preprocessing_threads=2)

    return hist


if __name__ == "__main__":
    hist = train()
