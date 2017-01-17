##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from keras.optimizers import SGD
from cnn.keras.models.deepROI4.model import build_model
from cnn.keras.deepROI4.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys

filter_length = int(sys.argv[1])

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_deepROI4_5'
path_model = '/home/mhubrich/checkpoints/adni/deepROI4_5/model.0099-loss_0.374-acc_0.946-val_loss_nan-val_acc_0.9504-val_fmeasure_0.9457-val_mcc_0.9009-val_mean_acc_0.9521.h5'


def mod3(a, b):
    c, x = divmod(a, b)
    z, y = divmod(c, b)
    return x, y, z


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/deepROI_fold1/5_test')
    groups, _ = sort_groups(scans_val)
    scans = groups['Normal'] + groups['AD']

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights(path_model)

    nb_runs = ((target_size[0]-filter_length+1) ** 3) + 1
    nb_pred = len(scans) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
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

    with open('predictions_deepROI4_5_fliter_' + str(filter_length) + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(scans)):
            for j in range(nb_runs):
                x, y, z = mod3(j, target_size[0] - filter_length + 1)
                writer.writerow([scans[i].imageID, scans[i].group, str(x), str(y), str(z), str(predictions[i*nb_runs + j, 0])])


if __name__ == "__main__":
    predict()

