##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras import callbacks
from cnn.keras.evaluation_callback2 import Evaluation
from keras.optimizers import SGD
from cnn.keras.models.deepROI4.model_merged import build_model
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
num_epoch = 2000
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = 'importanceMap_1_35_fold_' + fold + '_'
path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI7_22_CV' + fold


def load_data(scans):
    importanceMap_NC = np.load(path_importanceMap + 'NC.npy')
    importanceMap_AD = np.load(path_importanceMap + 'AD.npy')
    importanceMap_NC[np.where(importanceMap_NC <= 0)] = 0
    importanceMap_NC[np.where(importanceMap_NC > 0)] = 1
    importanceMap_AD[np.where(importanceMap_AD <= 0)] = 0
    importanceMap_AD[np.where(importanceMap_AD > 0)] = 1
    groups, _ = sort_groups(scans)
    nb_samples = 0
    for c in classes:
        assert groups[c] is not None, \
            'Could not find class %s' % c
        nb_samples += len(groups[c])
    X_NC = np.zeros((nb_samples, 1,) + target_size, dtype=np.float32)
    X_AD = np.zeros((nb_samples, 1,) + target_size, dtype=np.float32)
    y = np.zeros(nb_samples, dtype=np.int32)
    i = 0
    for c in classes:
        for scan in groups[c]:
            s = np.load(scan.path)
            X_NC[i] = s * importanceMap_NC
            X_AD[i] = s * importanceMap_AD
            y[i] = 0 if scan.group == classes[0] else 1
            i += 1
    return X_NC, X_AD, y


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_train')
    x_train_NC, x_train_AD, y_train = load_data(scans_train)

    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    x_val_NC, x_val_AD, y_val = load_data(scans_val)

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation([x_val_NC, x_val_AD], y_val, batch_size,
                       [callbacks.early_stop(patience=60, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=2, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc'])])]

    g, _ = sort_groups(scans_train)

    hist = model.fit(x=[x_train_NC, x_train_AD],
                     y=y_train,
                     nb_epoch=num_epoch,
                     callbacks=cbks,
                     class_weight={0:max(len(g['Normal']), len(g['AD']))/float(len(g['Normal'])),
                                   1:max(len(g['Normal']), len(g['AD']))/float(len(g['AD']))},
                     batch_size=batch_size,
                     shuffle=True,
                     verbose=2)


if __name__ == "__main__":
    train()

