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
from cnn.keras.models.AVG444.model4 import build_model
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])
pretrained = int(sys.argv[2])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
num_epoch = 5000
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_avgpool444_new'
path_checkpoints = '/home/mhubrich/checkpoints/adni/full_scan_25_rui_' + str(pretrained) + '_CV' + fold
path_model = ['/home/mhubrich/checkpoints/adni/full_scan_25_CV1/model.0076-loss_0.537-acc_0.889-val_loss_0.2799-val_acc_0.8806-val_fmeasure_0.8571-val_mcc_0.7583-val_mean_acc_0.8849.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV2/model.0173-loss_0.362-acc_0.949-val_loss_0.2189-val_acc_0.9313-val_fmeasure_0.9291-val_mcc_0.8634-val_mean_acc_0.9311.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV3/model.0164-loss_0.404-acc_0.933-val_loss_0.1093-val_acc_0.9750-val_fmeasure_0.9720-val_mcc_0.9495-val_mean_acc_0.9739.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV4/model.0076-loss_0.541-acc_0.882-val_loss_0.2093-val_acc_0.9416-val_fmeasure_0.9322-val_mcc_0.8809-val_mean_acc_0.9405.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV5/model.0179-loss_0.388-acc_0.938-val_loss_0.2337-val_acc_0.9172-val_fmeasure_0.8909-val_mcc_0.8246-val_mean_acc_0.9152.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV6/model.0129-loss_0.428-acc_0.922-val_loss_0.1566-val_acc_0.9366-val_fmeasure_0.9217-val_mcc_0.8694-val_mean_acc_0.9383.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV7/model.0094-loss_0.483-acc_0.904-val_loss_0.2861-val_acc_0.8913-val_fmeasure_0.8718-val_mcc_0.7799-val_mean_acc_0.8944.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV8/model.0068-loss_0.628-acc_0.848-val_loss_0.3229-val_acc_0.8980-val_fmeasure_0.8819-val_mcc_0.7928-val_mean_acc_0.8987.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV9/model.0109-loss_0.573-acc_0.876-val_loss_0.1884-val_acc_0.9470-val_fmeasure_0.9358-val_mcc_0.8907-val_mean_acc_0.9466.h5', '/home/mhubrich/checkpoints/adni/full_scan_25_CV10/model.0174-loss_0.349-acc_0.955-val_loss_0.1877-val_acc_0.9329-val_fmeasure_0.9123-val_mcc_0.8583-val_mean_acc_0.9320.h5']


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
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + fold + '_train')
    x_train, y_train = load_data(scans_train, flip=True)

    scans_val = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + fold + '_val')
    x_val, y_val = load_data(scans_val, flip=False)

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights(path_model[0])

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(x_val, y_val, batch_size,
                       [callbacks.early_stop(patience=40, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=2, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc'])])]

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

