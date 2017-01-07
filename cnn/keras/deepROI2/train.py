##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from keras.models import load_model
from cnn.keras import callbacks
from cnn.keras.evaluation_callback import Evaluation
from keras.optimizers import SGD
from cnn.keras.deepROI2.model import build_model
from cnn.keras.deepROI2.image_processing import inputs
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])
#sys.stdout = sys.stderr = open('output_1_' + fold, 'w')

# Training specific parameters
target_size = (20, 20, 20)
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = True
num_epoch = 1000
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_deepROI2_NC'
path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI2_merged_2_CV' + fold
path_weights = None

mod_NC = ['/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV1/model.0125-loss_0.325-acc_0.928-val_loss_0.4316-val_acc_0.8865-val_fmeasure_0.8824-val_mcc_0.7742-val_mean_acc_0.8879.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV2/model.0083-loss_0.368-acc_0.913-val_loss_0.3710-val_acc_0.8693-val_fmeasure_0.8684-val_mcc_0.7460-val_mean_acc_0.8733.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV3/model.0133-loss_0.343-acc_0.927-val_loss_0.3097-val_acc_0.9130-val_fmeasure_0.9091-val_mcc_0.8258-val_mean_acc_0.9129.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV4/model.0239-loss_0.202-acc_0.975-val_loss_0.2581-val_acc_0.9371-val_fmeasure_0.9367-val_mcc_0.8745-val_mean_acc_0.9372.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV5/model.0198-loss_0.221-acc_0.966-val_loss_0.5751-val_acc_0.9021-val_fmeasure_0.8923-val_mcc_0.8058-val_mean_acc_0.8998.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV6/model.0083-loss_0.350-acc_0.914-val_loss_0.2941-val_acc_0.9184-val_fmeasure_0.9200-val_mcc_0.8397-val_mean_acc_0.9206.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV7/model.0094-loss_0.364-acc_0.922-val_loss_0.1889-val_acc_0.9463-val_fmeasure_0.9459-val_mcc_0.8940-val_mean_acc_0.9468.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV8/model.0201-loss_0.221-acc_0.969-val_loss_0.2773-val_acc_0.9297-val_fmeasure_0.9204-val_mcc_0.8575-val_mean_acc_0.9296.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV9/model.0150-loss_0.296-acc_0.938-val_loss_0.2469-val_acc_0.9338-val_fmeasure_0.9343-val_mcc_0.8702-val_mean_acc_0.9353.h5', '/home/mhubrich/checkpoints/adni/deepROI2_NC_1_CV10/model.0202-loss_0.238-acc_0.963-val_loss_0.1520-val_acc_0.9688-val_fmeasure_0.9689-val_mcc_0.9394-val_mean_acc_0.9695.h5']
mod_AD = ['/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV1/model.0224-loss_0.210-acc_0.969-val_loss_0.4782-val_acc_0.9007-val_fmeasure_0.8955-val_mcc_0.8042-val_mean_acc_0.9038.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV2/model.0088-loss_0.429-acc_0.883-val_loss_0.3423-val_acc_0.8824-val_fmeasure_0.8846-val_mcc_0.7673-val_mean_acc_0.8831.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV3/model.0252-loss_0.259-acc_0.964-val_loss_0.3429-val_acc_0.9203-val_fmeasure_0.9220-val_mcc_0.8484-val_mean_acc_0.9254.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV4/model.0180-loss_0.254-acc_0.958-val_loss_0.2751-val_acc_0.9371-val_fmeasure_0.9383-val_mcc_0.8768-val_mean_acc_0.9390.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV5/model.0089-loss_0.424-acc_0.890-val_loss_0.3584-val_acc_0.8881-val_fmeasure_0.8710-val_mcc_0.7722-val_mean_acc_0.8861.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV6/model.0133-loss_0.327-acc_0.933-val_loss_0.2713-val_acc_0.9252-val_fmeasure_0.9252-val_mcc_0.8511-val_mean_acc_0.9256.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV7/model.0157-loss_0.288-acc_0.948-val_loss_0.2617-val_acc_0.9195-val_fmeasure_0.9200-val_mcc_0.8419-val_mean_acc_0.9212.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV8/model.0102-loss_0.427-acc_0.902-val_loss_0.2447-val_acc_0.9375-val_fmeasure_0.9322-val_mcc_0.8761-val_mean_acc_0.9359.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV9/model.0171-loss_0.295-acc_0.950-val_loss_0.2103-val_acc_0.9485-val_fmeasure_0.9481-val_mcc_0.8980-val_mean_acc_0.9488.h5', '/home/mhubrich/checkpoints/adni/deepROI2_AD_1_CV10/model.0191-loss_0.256-acc_0.958-val_loss_0.2395-val_acc_0.9375-val_fmeasure_0.9390-val_mcc_0.8752-val_mean_acc_0.9373.h5']
mod = ['/home/mhubrich/checkpoints/adni/AVG444_1_CV1/model.0357-loss_0.309-acc_0.966-val_loss_0.4842-val_acc_0.8794-val_fmeasure_0.8794-val_mcc_0.7590-val_mean_acc_0.8795.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV2/model.0159-loss_0.550-acc_0.878-val_loss_0.3468-val_acc_0.8889-val_fmeasure_0.8931-val_mcc_0.7781-val_mean_acc_0.8885.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV3/model.0440-loss_0.488-acc_0.901-val_loss_0.2656-val_acc_0.9275-val_fmeasure_0.9265-val_mcc_0.8566-val_mean_acc_0.9279.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV4/model.0297-loss_0.395-acc_0.936-val_loss_0.2155-val_acc_0.9434-val_fmeasure_0.9441-val_mcc_0.8886-val_mean_acc_0.9447.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV5/model.0241-loss_0.433-acc_0.922-val_loss_0.3891-val_acc_0.8811-val_fmeasure_0.8661-val_mcc_0.7600-val_mean_acc_0.8782.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV6/model.0184-loss_0.486-acc_0.904-val_loss_0.2749-val_acc_0.9184-val_fmeasure_0.9189-val_mcc_0.8381-val_mean_acc_0.9192.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV7/model.0232-loss_0.454-acc_0.913-val_loss_0.1351-val_acc_0.9664-val_fmeasure_0.9660-val_mcc_0.9337-val_mean_acc_0.9666.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV8/model.0187-loss_0.508-acc_0.902-val_loss_0.2162-val_acc_0.9375-val_fmeasure_0.9273-val_mcc_0.8744-val_mean_acc_0.9411.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV9/model.0274-loss_0.431-acc_0.924-val_loss_0.2381-val_acc_0.9485-val_fmeasure_0.9489-val_mcc_0.8996-val_mean_acc_0.9501.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV10/model.0356-loss_0.382-acc_0.937-val_loss_0.1543-val_acc_0.9688-val_fmeasure_0.9689-val_mcc_0.9394-val_mean_acc_0.9695.h5']

def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED, 'binary')
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'predict', SEED, 'binary')

    # Set up the model
    if path_weights is None:
        model = build_model()
        sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        model = load_model(path_weights)

    model.load_weights(mod_NC[int(fold)-1], by_name=True)
    model.load_weights(mod_AD[int(fold)-1], by_name=True)
    model.load_weights(mod[int(fold)-1], by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(val_inputs,
                       [callbacks.early_stop(patience=50, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=2, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc'])])]

    g, _ = sort_groups(scans_train)

    # Start training
    hist = model.fit_generator(
        train_inputs,
        samples_per_epoch=train_inputs.nb_sample,
        nb_epoch=num_epoch,
        callbacks=cbks,
        class_weight={0:max(len(g['Normal']), len(g['AD']))/float(len(g['Normal'])),
                      1:max(len(g['Normal']), len(g['AD']))/float(len(g['AD']))},
        verbose=2,
        max_q_size=32,
        nb_worker=1,
        pickle_safe=True)


if __name__ == "__main__":
    train()

