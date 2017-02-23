##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

import nibabel as nib
from cnn.keras import callbacks
from cnn.keras.evaluation_callback2 import Evaluation
from keras.optimizers import SGD
from cnn.keras.final_metaROI.model import build_model
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])

# Training specific parameters
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
num_epoch = 2000
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC'
path_checkpoints = '/home/mhubrich/checkpoints/adni/final_metaROI_merged_7_pSMC_CV' + fold


ranges = [(range(65, 73), range(28, 36), range(51, 59)),
          (range(73, 79), range(39, 45), range(27, 33)),
          (range(39, 49), range(37, 47), range(46, 56)),
          (range(13, 26), range(28, 41), range(47, 60)),
          (range(11, 16), range(35, 40), range(31, 36))]

def load_data(scans, flip=False):
    groups, _ = sort_groups(scans)
    nb_samples = 0
    for c in classes:
        assert groups[c] is not None, \
            'Could not find class %s' % c
        nb_samples += len(groups[c])
    if flip:
        nb_samples *= 2
    y = np.zeros(nb_samples, dtype=np.int32)
    X1 = np.zeros((nb_samples, 1, 8, 8, 8), dtype=np.float32)
    X2 = np.zeros((nb_samples, 1, 6, 6, 6), dtype=np.float32)
    X3 = np.zeros((nb_samples, 1, 10, 10, 10), dtype=np.float32)
    X4 = np.zeros((nb_samples, 1, 13, 13, 13), dtype=np.float32)
    X5 = np.zeros((nb_samples, 1, 5, 5, 5), dtype=np.float32)
    X = [X1, X2, X3, X4, X5]
    i = 0
    for c in classes:
        for scan in groups[c]:
            s = nib.load(scan.path).get_data()
            y[i] = 0 if scan.group == classes[0] else 1
            for j in range(len(X)):
                X[j][i] = s[ranges[j][0],:,:][:,ranges[j][1],:][:,:,ranges[j][2]]
            i += 1
            if flip:
                for j in range(len(X)):
                    X[j][i] = np.flipud(s)[ranges[j][0],:,:][:,ranges[j][1],:][:,:,ranges[j][2]]
                y[i] = y[i-1]
                i += 1
            del s
    return X, y


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_norui/' + fold + '_train', 'nii')
    x_train, y_train = load_data(scans_train, flip=True)

    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_norui/' + fold + '_val', 'nii')
    x_val, y_val = load_data(scans_val, flip=False)

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(x_val, y_val, batch_size,
                       [callbacks.early_stop(patience=150, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=1, monitor=['val_loss', 'val_acc', 'val_mean_acc'])])]

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

