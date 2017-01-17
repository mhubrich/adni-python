##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras.models.deepROI4.model import build_model
from cnn.keras.deepROI7.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys

# filter_length should be odd
filter_length = int(sys.argv[1])
fold = str(sys.argv[2])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = None
path_model = ['/home/mhubrich/checkpoints/adni/full_scan_28_CV1/model.0133-loss_0.443-acc_0.921-val_loss_0.3744-val_acc_0.8652-val_fmeasure_0.8571-val_mcc_0.7339-val_mean_acc_0.8690.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV2/model.0104-loss_0.501-acc_0.907-val_loss_0.3300-val_acc_0.8758-val_fmeasure_0.8742-val_mcc_0.7607-val_mean_acc_0.8811.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV3/model.0248-loss_0.438-acc_0.926-val_loss_0.2770-val_acc_0.9058-val_fmeasure_0.9008-val_mcc_0.8112-val_mean_acc_0.9059.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV4/model.0176-loss_0.409-acc_0.927-val_loss_0.1897-val_acc_0.9371-val_fmeasure_0.9367-val_mcc_0.8745-val_mean_acc_0.9372.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV5/model.0092-loss_0.559-acc_0.877-val_loss_0.3278-val_acc_0.8881-val_fmeasure_0.8689-val_mcc_0.7717-val_mean_acc_0.8874.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV6/model.0106-loss_0.443-acc_0.929-val_loss_0.2538-val_acc_0.9184-val_fmeasure_0.9155-val_mcc_0.8369-val_mean_acc_0.9188.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV7/model.0126-loss_0.459-acc_0.910-val_loss_0.1692-val_acc_0.9329-val_fmeasure_0.9296-val_mcc_0.8658-val_mean_acc_0.9335.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV8/model.0142-loss_0.413-acc_0.943-val_loss_0.1966-val_acc_0.9375-val_fmeasure_0.9298-val_mcc_0.8735-val_mean_acc_0.9367.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV9/model.0285-loss_0.301-acc_0.971-val_loss_0.1388-val_acc_0.9485-val_fmeasure_0.9481-val_mcc_0.8980-val_mean_acc_0.9488.h5', '/home/mhubrich/checkpoints/adni/full_scan_28_CV10/model.0144-loss_0.435-acc_0.927-val_loss_0.1678-val_acc_0.9563-val_fmeasure_0.9565-val_mcc_0.9144-val_mean_acc_0.9570.h5']


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights(path_model[int(fold) - 1])

    if path_importanceMap is None:
        importanceMap = np.ones(target_size)
    else:
        importanceMap = np.load(path_importanceMap)
        importanceMap[np.where(importanceMap <= 0)] = 0
        importanceMap[np.where(importanceMap > 0)] = 1
    indices = np.where(importanceMap > 0)
    nb_runs = len(indices[0]) + 1
    nb_pred = len(scans) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
    i = 0
    for scan in scans:
        val_inputs = ScanIterator(scan, path_importanceMap, filter_length, target_size, batch_size=batch_size, seed=SEED)
        predictions[i:i+val_inputs.nb_sample] = model.predict_generator(val_inputs,
                                                                        val_inputs.nb_sample,
                                                                        max_q_size=batch_size,
                                                                        nb_worker=1,
                                                                        pickle_safe=True)
        i += val_inputs.nb_sample
        del val_inputs

    with open('predictions_deepROI7_1_fliter_' + str(filter_length) + '_fold_' + fold + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(scans)):
            for j in range(nb_runs):
                if j < len(indices[0]):
                    x, y, z = indices[0][j], indices[1][j], indices[2][j]
                else:
                    x, y, z = -1, -1, -1
                writer.writerow([scans[i].imageID, scans[i].group, str(x), str(y), str(z), str(predictions[i*nb_runs + j, 0])])


if __name__ == "__main__":
    predict()

