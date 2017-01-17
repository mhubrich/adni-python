##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from keras.optimizers import SGD
from cnn.keras.models.AVG444.model4 import build_model
from cnn.keras.AVG444.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys

fold = str(sys.argv[1])
filter_length = int(sys.argv[2])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_avgpool444_new'
path_model = ['/home/mhubrich/checkpoints/adni/full_scan_24_CV1/model.0099-loss_0.266-acc_0.978-val_loss_nan-val_acc_0.8936-val_fmeasure_0.8889-val_mcc_0.7891-val_mean_acc_0.8957.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV2/model.0131-loss_0.268-acc_0.976-val_loss_nan-val_acc_0.9085-val_fmeasure_0.9079-val_mcc_0.8248-val_mean_acc_0.9127.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV3/model.0054-loss_0.430-acc_0.929-val_loss_0.3100-val_acc_0.9203-val_fmeasure_0.9185-val_mcc_0.8414-val_mean_acc_0.9203.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV4/model.0044-loss_0.439-acc_0.918-val_loss_nan-val_acc_0.9434-val_fmeasure_0.9441-val_mcc_0.8886-val_mean_acc_0.9447.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV5/model.0043-loss_0.432-acc_0.920-val_loss_0.3293-val_acc_0.9091-val_fmeasure_0.8960-val_mcc_0.8153-val_mean_acc_0.9069.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV6/model.0037-loss_0.434-acc_0.928-val_loss_0.2709-val_acc_0.9184-val_fmeasure_0.9189-val_mcc_0.8381-val_mean_acc_0.9192.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV7/model.0024-loss_0.525-acc_0.883-val_loss_0.1735-val_acc_0.9664-val_fmeasure_0.9660-val_mcc_0.9337-val_mean_acc_0.9666.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV8/model.0030-loss_0.495-acc_0.904-val_loss_0.2044-val_acc_0.9453-val_fmeasure_0.9381-val_mcc_0.8892-val_mean_acc_0.9454.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV9/model.0066-loss_0.368-acc_0.945-val_loss_0.1994-val_acc_0.9412-val_fmeasure_0.9412-val_mcc_0.8840-val_mean_acc_0.9420.h5', '/home/mhubrich/checkpoints/adni/full_scan_24_CV10/model.0035-loss_0.446-acc_0.920-val_loss_0.1727-val_acc_0.9688-val_fmeasure_0.9689-val_mcc_0.9394-val_mean_acc_0.9695.h5']


def mod3(a, b):
    c, x = divmod(a, b)
    z, y = divmod(c, b)
    return x, y, z


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights(path_model[int(fold)-1])

    nb_runs = ((target_size[0]-filter_length+1) ** 3) + 1
    nb_pred = len(scans_val) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']
    i = 0
    for scan in scans:
        val_inputs = ScanIterator(scan, None, target_size, load_all_scans, classes=classes, class_mode=None, batch_size=batch_size, shuffle=False, seed=SEED, filter_length=filter_length)
        predictions[i:i+val_inputs.nb_sample] = model.predict_generator(val_inputs,
                                                                        val_inputs.nb_sample,
                                                                        max_q_size=batch_size,
                                                                        nb_worker=1,
                                                                        pickle_safe=True)
        i += val_inputs.nb_sample
        del val_inputs

    with open('predictions_model4_fliter_' + str(filter_length) + '_CV' + fold + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(scans)):
            for j in range(nb_runs):
                x, y, z = mod3(j, target_size[0] - filter_length + 1)
                writer.writerow([scans[i].imageID, str(x), str(y), str(z), str(predictions[i*nb_runs + j, 0])])


if __name__ == "__main__":
    predict()

