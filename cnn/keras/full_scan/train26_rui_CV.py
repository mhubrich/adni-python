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
path_ADNI = '/home/mhubrich/ADNI_pSMC_avgpool444'
path_checkpoints = '/home/mhubrich/checkpoints/adni/full_scan_26_rui_' + str(pretrained) + '_CV' + fold
path_model = ['/home/mhubrich/checkpoints/adni/full_scan_26_CV1/model.0101-loss_0.605-acc_0.860-val_loss_0.2701-val_acc_0.8955-val_fmeasure_0.8772-val_mcc_0.7879-val_mean_acc_0.8976.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV2/model.0320-loss_0.363-acc_0.943-val_loss_0.1778-val_acc_0.9313-val_fmeasure_0.9256-val_mcc_0.8628-val_mean_acc_0.9329.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV3/model.0252-loss_0.355-acc_0.953-val_loss_0.1101-val_acc_0.9750-val_fmeasure_0.9720-val_mcc_0.9495-val_mean_acc_0.9739.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV4/model.0324-loss_0.311-acc_0.958-val_loss_0.1180-val_acc_0.9635-val_fmeasure_0.9573-val_mcc_0.9255-val_mean_acc_0.9638.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV5/model.0240-loss_0.412-acc_0.932-val_loss_0.2070-val_acc_0.9379-val_fmeasure_0.9189-val_mcc_0.8687-val_mean_acc_0.9359.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV6/model.0225-loss_0.435-acc_0.917-val_loss_0.1442-val_acc_0.9577-val_fmeasure_0.9492-val_mcc_0.9130-val_mean_acc_0.9565.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV7/model.0160-loss_0.427-acc_0.918-val_loss_0.3011-val_acc_0.8986-val_fmeasure_0.8814-val_mcc_0.7943-val_mean_acc_0.9006.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV8/model.0112-loss_0.560-acc_0.882-val_loss_0.3510-val_acc_0.8639-val_fmeasure_0.8438-val_mcc_0.7236-val_mean_acc_0.8631.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV9/model.0294-loss_0.523-acc_0.890-val_loss_0.1913-val_acc_0.9545-val_fmeasure_0.9444-val_mcc_0.9065-val_mean_acc_0.9558.h5', '/home/mhubrich/checkpoints/adni/full_scan_26_CV10/model.0231-loss_0.377-acc_0.944-val_loss_0.2015-val_acc_0.9396-val_fmeasure_0.9204-val_mcc_0.8726-val_mean_acc_0.9408.h5']


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

