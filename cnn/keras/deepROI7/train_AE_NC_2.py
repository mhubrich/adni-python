##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

import os
from cnn.keras import callbacks
from cnn.keras.evaluation_callback2 import Evaluation
from keras.optimizers import SGD
from cnn.keras.models.deepROI4.model_AE import build_model
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal']
batch_size = 32
num_epoch = 5000
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = 'importanceMap_1_35_fold_' + fold + '_'
path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI7_AE_NC_2_CV' + fold


def load_data(scans):
    importanceMap_NC = np.load(path_importanceMap + 'NC.npy')
    importanceMap_NC[np.where(importanceMap_NC <= 0)] = 0
    importanceMap_NC[np.where(importanceMap_NC > 0)] = 1
    groups, _ = sort_groups(scans)
    nb_samples = 0
    for c in classes:
        assert groups[c] is not None, \
            'Could not find class %s' % c
        nb_samples += len(groups[c])
    X_NC = np.zeros((nb_samples, 1,) + target_size, dtype=np.float32)
    y = np.zeros(nb_samples, dtype=np.int32)
    i = 0
    for c in classes:
        for scan in groups[c]:
            X_NC[i] = np.load(scan.path) * importanceMap_NC
            y[i] = 0 if scan.group == classes[0] else 1
            i += 1
    return X_NC, y


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_train')
    x_train_NC, y_train = load_data(scans_train)

    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    x_val_NC, y_val = load_data(scans_val)

    # Set up the model
    model = build_model()
    #sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    if not os.path.exists(path_checkpoints):
        os.makedirs(path_checkpoints)

    path_checkp = os.path.join(path_checkpoints, 'weights.h5')

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            callbacks.ModelCheckpoint(path_checkp, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto'),
            callbacks.EarlyStopping(monitor='val_loss', patience=80, verbose=0, mode='auto')]

    hist = model.fit(x=x_train_NC,
                     y=x_train_NC,
                     validation_data=(x_val_NC, x_val_NC),
                     nb_epoch=num_epoch,
                     callbacks=cbks,
                     batch_size=batch_size,
                     shuffle=True,
                     verbose=2)


if __name__ == "__main__":
    train()

