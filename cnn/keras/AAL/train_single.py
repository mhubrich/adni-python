from cnn.keras import callbacks
from keras.optimizers import SGD
from cnn.keras.models.AAL61.model_single9 import build_model
from cnn.keras.AAL.image_processing_single import inputs
from utils.split_scans import read_imageID
import sys
#sys.stdout = sys.stderr = open('output_diff_19', 'w')

fold = str(sys.argv[1])

# Training specific parameters
target_size = (19, 22, 11)
SEED = 0  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = False
num_epoch = 500
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_AAL61_new'
path_checkpoints = '/home/mhubrich/checkpoints/adni/AAL61_new_2_CV' + fold
path_weights = None


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV/' + fold + '_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV/' + fold + '_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'val', SEED)

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if path_weights:
        model.load_weights(path_weights)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            callbacks.early_stopping(max_acc=0.97, patience=10),
            callbacks.save_model(path_checkpoints, max_files=5)]

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

