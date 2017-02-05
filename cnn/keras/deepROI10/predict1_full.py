##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras.models.deepROI4.model import build_model
from cnn.keras.deepROI10.get_paths import get_paths
from cnn.keras.deepROI10.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys


# filter_length should be odd
filter_length = int(sys.argv[1])
# fold rui
fold = str(sys.argv[2])
# fold trained CNN
fold2 = str(sys.argv[3])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 96
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_model = get_paths(n=1, fold=int(fold2))


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + fold + '_train')
    scans_val += read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + fold + '_val')
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights(path_model)

    indices = np.where(np.ones(target_size) > 0)
    nb_runs = len(indices[0]) + 1
    nb_pred = len(scans) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
    i = 0
    for scan in scans:
        val_inputs = ScanIterator(scan, filter_length, target_size, batch_size=batch_size, seed=SEED)
        predictions[i:i+val_inputs.nb_sample] = model.predict_generator(val_inputs,
                                                                        val_inputs.nb_sample,
                                                                        max_q_size=batch_size,
                                                                        nb_worker=1,
                                                                        pickle_safe=True)
        i += val_inputs.nb_sample

    with open('predictions/predictions_deepROI10_2_fliter_' + str(filter_length) + '_fold_' + fold + '_' + fold2 + '_full.csv', 'wb') as csvfile:
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

