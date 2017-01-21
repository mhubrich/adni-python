##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from cnn.keras.models.deepROI4.model_merged import build_model
from cnn.keras.deepROI7.predict_iterator2 import ScanIterator
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
batch_size = 96
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = 'importanceMap_1_35_fold_' + fold + '_'
path_model = ['/home/mhubrich/checkpoints/adni/deepROI7_22_CV1/model.0463-loss_0.505-acc_0.941-val_loss_0.4313-val_acc_0.8865-val_fmeasure_0.8857-val_mcc_0.7730-val_mean_acc_0.8865.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV2/model.0400-loss_0.500-acc_0.941-val_loss_0.4167-val_acc_0.8889-val_fmeasure_0.8874-val_mcc_0.7870-val_mean_acc_0.8942.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV3/model.0271-loss_1.002-acc_0.769-val_loss_0.6012-val_acc_0.8623-val_fmeasure_0.8504-val_mcc_0.7252-val_mean_acc_0.8647.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV4/model.0353-loss_0.613-acc_0.893-val_loss_0.2331-val_acc_0.9371-val_fmeasure_0.9359-val_mcc_0.8742-val_mean_acc_0.9371.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV5/model.0296-loss_0.576-acc_0.907-val_loss_0.4216-val_acc_0.8741-val_fmeasure_0.8571-val_mcc_0.7450-val_mean_acc_0.8712.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV6/model.0306-loss_0.633-acc_0.891-val_loss_0.2874-val_acc_0.9184-val_fmeasure_0.9155-val_mcc_0.8369-val_mean_acc_0.9188.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV7/model.0365-loss_0.583-acc_0.905-val_loss_0.1574-val_acc_0.9664-val_fmeasure_0.9660-val_mcc_0.9337-val_mean_acc_0.9666.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV8/model.0357-loss_0.608-acc_0.903-val_loss_0.2149-val_acc_0.9375-val_fmeasure_0.9298-val_mcc_0.8735-val_mean_acc_0.9367.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV9/model.0438-loss_0.551-acc_0.915-val_loss_0.2520-val_acc_0.9338-val_fmeasure_0.9353-val_mcc_0.8725-val_mean_acc_0.9373.h5', '/home/mhubrich/checkpoints/adni/deepROI7_22_CV10/model.0452-loss_0.544-acc_0.923-val_loss_0.1753-val_acc_0.9625-val_fmeasure_0.9625-val_mcc_0.9277-val_mean_acc_0.9639.h5']


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.load_weights(path_model[int(fold) - 1])

    importanceMap_NC = np.load(path_importanceMap + 'NC.npy')
    importanceMap_NC[np.where(importanceMap_NC <= 0)] = 0
    importanceMap_NC[np.where(importanceMap_NC > 0)] = 1
    importanceMap_AD = np.load(path_importanceMap + 'AD.npy')
    importanceMap_AD[np.where(importanceMap_AD <= 0)] = 0
    importanceMap_AD[np.where(importanceMap_AD > 0)] = 1
    importanceMap = importanceMap_NC + importanceMap_AD
    importanceMap[np.where(importanceMap > 1)] = 1
    indices = np.where(importanceMap > 0)
    nb_runs = len(indices[0]) + 1
    nb_pred = len(scans) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
    i = 0
    for scan in scans:
        val_inputs = ScanIterator(scan, path_importanceMap + 'NC.npy', path_importanceMap + 'AD.npy', filter_length, target_size, batch_size=batch_size, seed=SEED)
        predictions[i:i+val_inputs.nb_sample] = model.predict_generator(val_inputs,
                                                                        val_inputs.nb_sample,
                                                                        max_q_size=batch_size,
                                                                        nb_worker=1,
                                                                        pickle_safe=True)
        i += val_inputs.nb_sample
        del val_inputs

    np.save('save_2_fliter_' + str(filter_length) + '_fold_' + fold + '.npy', predictions)
    with open('predictions_deepROI7_2_fliter_' + str(filter_length) + '_fold_' + fold + '.csv', 'wb') as csvfile:
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

