##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from keras.optimizers import SGD
from cnn.keras.models.AVG444.model_normal import build_model
from cnn.keras.AVG444.predict_iterator import ScanIterator
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import csv
import sys

fold = str(sys.argv[1])

# Training specific parameters
target_size = (22, 22, 22)
filter_length = 1
classes = ['MCI', 'AD']
batch_size = 128
load_all_scans = True
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_avgpool444_new'
path_model = ['/home/mhubrich/checkpoints/adni/full_scan_23_CV1/model.0032-loss_0.852-acc_0.767-val_loss_0.4420-val_acc_0.8132-val_fmeasure_0.6600-val_mcc_0.5592-val_mean_acc_0.8174.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV2/model.0090-loss_0.715-acc_0.820-val_loss_0.3535-val_acc_0.8717-val_fmeasure_0.8095-val_mcc_0.7127-val_mean_acc_0.8564.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV3/model.0068-loss_0.765-acc_0.797-val_loss_0.4198-val_acc_0.8377-val_fmeasure_0.7704-val_mcc_0.6513-val_mean_acc_0.8138.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV4/model.0054-loss_0.749-acc_0.791-val_loss_0.3315-val_acc_0.8676-val_fmeasure_0.7840-val_mcc_0.6900-val_mean_acc_0.8374.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV5/model.0088-loss_0.646-acc_0.850-val_loss_0.4533-val_acc_0.8324-val_fmeasure_0.6990-val_mcc_0.5863-val_mean_acc_0.8057.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV6/model.0141-loss_0.532-acc_0.894-val_loss_0.3618-val_acc_0.8811-val_fmeasure_0.8136-val_mcc_0.7263-val_mean_acc_0.8631.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV7/model.0019-loss_0.862-acc_0.761-val_loss_0.3697-val_acc_0.8470-val_fmeasure_0.7455-val_mcc_0.6363-val_mean_acc_0.8149.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV8/model.0051-loss_0.784-acc_0.780-val_loss_0.3900-val_acc_0.8659-val_fmeasure_0.7447-val_mcc_0.6657-val_mean_acc_0.8616.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV9/model.0138-loss_0.536-acc_0.885-val_loss_0.4759-val_acc_0.8525-val_fmeasure_0.7273-val_mcc_0.6262-val_mean_acc_0.8111.h5', '/home/mhubrich/checkpoints/adni/full_scan_23_CV10/model.0044-loss_0.803-acc_0.767-val_loss_0.3182-val_acc_0.8842-val_fmeasure_0.8136-val_mcc_0.7319-val_mean_acc_0.8558.h5']

def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_MCI/' + fold + '_val')

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights(path_model[int(fold)-1])

    nb_runs = ((target_size[0]-filter_length+1) ** 3) + 1
    nb_pred = len(scans_val) * nb_runs
    predictions = np.zeros((nb_pred, 1), dtype=np.float32)
    groups, _ = sort_groups(scans_val)
    scans = groups['MCI'] + groups['AD']
    i = 0
    for scan in scans:
        val_inputs = ScanIterator(scan, None, target_size, load_all_scans, classes=classes, class_mode=None, batch_size=batch_size, shuffle=False, seed=SEED, filter_length=filter_length)
        predictions[i:i+val_inputs.nb_sample] = model.predict_generator(val_inputs,
                                                                        val_inputs.nb_sample,
                                                                        max_q_size=4*batch_size,
                                                                        nb_worker=1,
                                                                        pickle_safe=True)
        i += val_inputs.nb_sample

    with open('predictions_MCI_fliter_' + str(filter_length) + '_CV' + fold + '.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(scans)):
            for j in range(nb_runs):
                x, y, z = val_inputs.mod3(j, target_size[0] - filter_length + 1)
                writer.writerow([scans[i].imageID, str(x), str(y), str(z), str(predictions[i*nb_runs + j, 0])])


if __name__ == "__main__":
    predict()

