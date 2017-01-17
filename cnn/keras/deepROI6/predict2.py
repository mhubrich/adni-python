##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras.models.deepROI4.model import build_model
from cnn.keras.deepROI6.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys

# filter_length should be odd
filter_length = int(sys.argv[1])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = 'importanceMap_1_35.npy'
path_model = '/home/mhubrich/checkpoints/adni/deepROI6_2/model.0288-loss_0.401-acc_0.937-val_loss_0.3985-val_acc_0.8943-val_fmeasure_0.8738-val_mcc_0.7840-val_mean_acc_0.8952.h5'


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/deepROI_fold1/2_test')
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights(path_model)

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

    with open('predictions_deepROI6_2_fliter_' + str(filter_length) + '.csv', 'wb') as csvfile:
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

