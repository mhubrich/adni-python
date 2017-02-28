##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras.final_full_scan.model import build_model
from cnn.keras.final_deepROI1.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys


# filter_length should be odd
filter_length = int(sys.argv[1])
# fold
fold = str(sys.argv[2])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 256
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_avgpool444_2'
path_model = ['/home/mhubrich/checkpoints/adni/final_full_scan_1_CV1/model.0086-loss_0.482-acc_0.908-val_loss_0.2359-val_acc_0.9265-val_mean_acc_0.9222.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV2/model.0075-loss_0.534-acc_0.892-val_loss_0.3067-val_acc_0.8812-val_mean_acc_0.8811.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV3/model.0222-loss_0.485-acc_0.906-val_loss_0.2494-val_acc_0.9176-val_mean_acc_0.9199.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV4/model.0149-loss_0.489-acc_0.904-val_loss_0.2230-val_acc_0.9321-val_mean_acc_0.9312.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV5/model.0059-loss_0.593-acc_0.863-val_loss_0.2863-val_acc_0.8864-val_mean_acc_0.8788.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV6/model.0129-loss_0.425-acc_0.933-val_loss_0.2240-val_acc_0.9275-val_mean_acc_0.9269.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV7/model.0186-loss_0.382-acc_0.946-val_loss_0.1558-val_acc_0.9570-val_mean_acc_0.9572.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV8/model.0212-loss_0.390-acc_0.939-val_loss_0.1951-val_acc_0.9331-val_mean_acc_0.9328.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV9/model.0144-loss_0.428-acc_0.924-val_loss_0.2374-val_acc_0.9041-val_mean_acc_0.9024.h5', '/home/mhubrich/checkpoints/adni/final_full_scan_1_CV10/model.0093-loss_0.442-acc_0.925-val_loss_0.1479-val_acc_0.9628-val_mean_acc_0.9627.h5']


def predict():
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean3/' + fold + '_train')
    groups, _ = sort_groups(scans_train)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights(path_model[int(fold)-1])

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

    with open('predictions/predictions_deepROI1_fliter_' + str(filter_length) + '_fold_' + fold + '.csv', 'wb') as csvfile:
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

