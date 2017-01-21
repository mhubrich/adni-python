##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras.models.deepROI4.model import build_model
from cnn.keras.deepROI8.predict_iterator2 import ScanIterator
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
path_model = ['/home/mhubrich/checkpoints/adni/deepROI8_1_CV1/model.0202-loss_0.581-acc_0.874-val_loss_0.2120-val_acc_0.9277-val_mean_acc_0.9297.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV2/model.0193-loss_0.562-acc_0.893-val_loss_0.2735-val_acc_0.8795-val_mean_acc_0.8823.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV3/model.0474-loss_0.300-acc_0.975-val_loss_0.2955-val_acc_0.9150-val_mean_acc_0.9139.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV4/model.0322-loss_0.442-acc_0.924-val_loss_0.2971-val_acc_0.8848-val_mean_acc_0.8868.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV5/model.0292-loss_0.372-acc_0.946-val_loss_0.2420-val_acc_0.9150-val_mean_acc_0.9158.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV6/model.0418-loss_0.301-acc_0.967-val_loss_0.1978-val_acc_0.9085-val_mean_acc_0.9042.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV7/model.0338-loss_0.385-acc_0.940-val_loss_0.3565-val_acc_0.8500-val_mean_acc_0.8503.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV8/model.0322-loss_0.413-acc_0.933-val_loss_0.2692-val_acc_0.9051-val_mean_acc_0.9051.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV9/model.0203-loss_0.518-acc_0.889-val_loss_0.4169-val_acc_0.8286-val_mean_acc_0.8314.h5', '/home/mhubrich/checkpoints/adni/deepROI8_1_CV10/model.0377-loss_0.382-acc_0.942-val_loss_0.1941-val_acc_0.9313-val_mean_acc_0.9310.h5']

def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights(path_model[int(fold) - 1])

    grid = np.zeros(target_size, dtype=np.int32)
    for x in range(0, target_size[0], 2):
        for y in range(0, target_size[1], 2):
            for z in range(0, target_size[2], 2):
                grid[x,y,z] = 1
    indices = np.where(grid > 0)
    nb_runs = len(indices[0]) + 1
    nb_pred = len(scans) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
    i = 0
    for scan in scans:
        val_inputs = ScanIterator(scan, grid, filter_length, target_size, batch_size=batch_size, seed=SEED)
        predictions[i:i+val_inputs.nb_sample] = model.predict_generator(val_inputs,
                                                                        val_inputs.nb_sample,
                                                                        max_q_size=batch_size,
                                                                        nb_worker=1,
                                                                        pickle_safe=True)
        i += val_inputs.nb_sample

    with open('predictions_deepROI8_1_fliter_' + str(filter_length) + '_fold_' + fold + '.csv', 'wb') as csvfile:
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

