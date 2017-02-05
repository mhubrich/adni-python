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
from cnn.keras.models.deepROI4.model import build_model
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys


fold = str(sys.argv[1])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
num_epoch = 1500
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_avgpool444_2'
path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI12_4_CV' + fold


def load_data(scans, flip=False):
    groups, _ = sort_groups(scans)
    nb_samples = 0
    for c in classes:
        assert groups[c] is not None, \
            'Could not find class %s' % c
        nb_samples += len(groups[c])
    if flip:
        X = np.zeros((2*nb_samples, 1, ) + target_size, dtype=np.float32)
        y = np.zeros(2*nb_samples, dtype=np.int32)
    else:
        X = np.zeros((nb_samples, 1, ) + target_size, dtype=np.float32)
        y = np.zeros(nb_samples, dtype=np.int32)
    i = 0
    for c in classes:
        for scan in groups[c]:
            X[i] = np.load(scan.path)
            y[i] = 0 if scan.group == classes[0] else 1
            i += 1
            if flip:
                X[i] = np.flipud(X[i-1, 0])
                y[i] = y[i-1]
                i += 1
    return X, y

def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_deepROI12/' + fold + '_train')
    x_train, y_train = load_data(scans_train, flip=True)

    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_deepROI12/' + fold + '_val')
    x_val, y_val = load_data(scans_val)

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(x_val, y_val, batch_size,
                       [callbacks.early_stop(patience=60, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=1, monitor=['val_loss', 'val_acc'])])]

    g, _ = sort_groups(scans_train)

    hist = model.fit(x=x_train,
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

