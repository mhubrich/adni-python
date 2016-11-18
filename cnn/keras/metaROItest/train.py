from keras.optimizers import SGD
from cnn.keras import callbacks
from cnn.keras.metaROItest.model import build_model
from cnn.keras.metaROItest.image_processing import inputs
from utils.split_scans import read_imageID
import sys
sys.stdout = stderr = open('output_pretrained_test_norot_2', 'w')


# Training specific parameters
target_size = (6, 6, 6)
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
num_epoch = 9999
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_metaROI1'
path_checkpoints = '/home/mhubrich/checkpoints/adni/metaROI_pretrained_test_norot_2'
path_weights = None


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/test_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/test_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'val', SEED)

    # Set up the model
    model = build_model(num_classes=len(classes))
    sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if path_weights:
        model.load_weights(path_weights)
    model.load_weights('/home/mhubrich/checkpoints/adni/metaROI1_test_2/model.2836-loss_0.555-acc_0.786-val_loss_0.4557-val_acc_0.8109.h5', by_name=True)
    model.load_weights('/home/mhubrich/checkpoints/adni/metaROI2_test_2/model.1657-loss_0.542-acc_0.781-val_loss_0.4834-val_acc_0.8073.h5', by_name=True)
    model.load_weights('/home/mhubrich/checkpoints/adni/metaROI2_test_3/model.2371-loss_0.500-acc_0.844-val_loss_0.4060-val_acc_0.8618.h5', by_name=True)
    model.load_weights('/home/mhubrich/checkpoints/adni/metaROI4_test_2/model.2880-loss_0.539-acc_0.823-val_loss_0.5482-val_acc_0.8000.h5', by_name=True)
    model.load_weights('/home/mhubrich/checkpoints/adni/metaROI5_test_2/model.4460-loss_0.538-acc_0.775-val_loss_0.4771-val_acc_0.7927.h5', by_name=True)
    model.load_weights('/home/mhubrich/checkpoints/adni/metaROI_diff_test_1/model.7929-loss_0.416-acc_0.812-val_loss_0.3496-val_acc_0.8545.h5', by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            callbacks.save_model(path_checkpoints, max_files=3)]

    # Start training
    model.fit_generator(
        train_inputs,
        samples_per_epoch=train_inputs.nb_sample,
        nb_epoch=num_epoch,
        validation_data=val_inputs,
        nb_val_samples=val_inputs.nb_sample,
        callbacks=cbks,
        verbose=2,
        max_q_size=256,
        nb_worker=1,
        pickle_safe=True)


if __name__ == "__main__":
    train()

