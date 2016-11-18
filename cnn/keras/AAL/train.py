from cnn.keras import callbacks
from keras.optimizers import SGD
from cnn.keras.AAL.model import build_model
from cnn.keras.AAL.image_processing import inputs
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys
sys.stdout = sys.stderr = open('output_test_10', 'w')


# Training specific parameters
target_size = (18, 18, 18)
SEED = 42  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 64
load_all_scans = True
num_epoch = 1000
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_AAL64'
path_checkpoints = '/home/mhubrich/checkpoints/adni/AAL_test_10'
path_weights = None


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/train_intnorm')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/val_intnorm')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'val', SEED)

    # Set up the model
    model = build_model(2)
    sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if path_weights:
        model.load_weights(path_weights)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL64_55/model.0259-loss_0.497-acc_0.825-val_loss_0.3369-val_acc_0.8827.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL65_7/model.0116-loss_0.651-acc_0.734-val_loss_0.4064-val_acc_0.8783.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL34_8/model.0492-loss_0.373-acc_0.865-val_loss_0.2789-val_acc_0.9115.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL35_9/model.0540-loss_0.335-acc_0.888-val_loss_0.2476-val_acc_0.9071.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL61_8/model.0156-loss_0.543-acc_0.787-val_loss_0.2705-val_acc_0.9071.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL_diff_2/model.4763-loss_0.425-acc_0.819-val_loss_0.3649-val_acc_0.8916.h5', by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            callbacks.early_stopping(max_acc=0.95, patience=5),
            callbacks.save_model(path_checkpoints, max_files=3)]

    g, _ = sort_groups(scans_train)

    # Start training
    model.fit_generator(
        train_inputs,
        samples_per_epoch=train_inputs.nb_sample,
        nb_epoch=num_epoch,
        validation_data=val_inputs,
        nb_val_samples=val_inputs.nb_sample,
        callbacks=cbks,
        #class_weight={0:max(len(g['Normal']), len(g['AD']))/float(len(g['Normal'])),
        #              1:max(len(g['Normal']), len(g['AD']))/float(len(g['AD']))},
        verbose=2,
        max_q_size=128,
        nb_worker=1,
        pickle_safe=True)


if __name__ == "__main__":
    train()

