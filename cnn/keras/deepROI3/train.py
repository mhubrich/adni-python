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
from cnn.keras.deepROI3.model import build_model
from cnn.keras.deepROI3.image_processing import inputs
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
path_ADNI = '/home/mhubrich/ADNI_intnorm_deepROI3_NC'
path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI3_merged_1_CV' + fold
path_weights = None

mod_NC = ['/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV1/model.0073-loss_0.456-acc_0.892-val_loss_0.3814-val_acc_0.8652-val_fmeasure_0.8593-val_mcc_0.7322-val_mean_acc_0.8672.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV2/model.0055-loss_0.521-acc_0.861-val_loss_0.3570-val_acc_0.8366-val_fmeasure_0.8408-val_mcc_0.6748-val_mean_acc_0.8368.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV3/model.0210-loss_0.426-acc_0.901-val_loss_0.2897-val_acc_0.9058-val_fmeasure_0.8992-val_mcc_0.8117-val_mean_acc_0.9070.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV4/model.0189-loss_0.290-acc_0.954-val_loss_0.2306-val_acc_0.9308-val_fmeasure_0.9299-val_mcc_0.8617-val_mean_acc_0.9308.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV5/model.0047-loss_0.557-acc_0.843-val_loss_0.3662-val_acc_0.8671-val_fmeasure_0.8504-val_mcc_0.7317-val_mean_acc_0.8641.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV6/model.0044-loss_0.599-acc_0.836-val_loss_0.2956-val_acc_0.8844-val_fmeasure_0.8811-val_mcc_0.7686-val_mean_acc_0.8845.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV7/model.0114-loss_0.396-acc_0.914-val_loss_0.1833-val_acc_0.9262-val_fmeasure_0.9241-val_mcc_0.8523-val_mean_acc_0.9260.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV8/model.0224-loss_0.273-acc_0.958-val_loss_0.2384-val_acc_0.9297-val_fmeasure_0.9204-val_mcc_0.8575-val_mean_acc_0.9296.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV9/model.0115-loss_0.389-acc_0.917-val_loss_0.2541-val_acc_0.9191-val_fmeasure_0.9173-val_mcc_0.8383-val_mean_acc_0.9190.h5', '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV10/model.0182-loss_0.319-acc_0.943-val_loss_0.1708-val_acc_0.9563-val_fmeasure_0.9560-val_mcc_0.9162-val_mean_acc_0.9583.h5']
mod_AD = ['/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV1/model.0235-loss_0.284-acc_0.958-val_loss_0.4296-val_acc_0.9007-val_fmeasure_0.8923-val_mcc_0.8094-val_mean_acc_0.9093.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV2/model.0124-loss_0.406-acc_0.908-val_loss_0.4319-val_acc_0.8627-val_fmeasure_0.8727-val_mcc_0.7245-val_mean_acc_0.8634.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV3/model.0226-loss_0.361-acc_0.927-val_loss_0.3035-val_acc_0.9203-val_fmeasure_0.9185-val_mcc_0.8414-val_mean_acc_0.9203.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV4/model.0181-loss_0.353-acc_0.930-val_loss_0.2496-val_acc_0.9308-val_fmeasure_0.9299-val_mcc_0.8617-val_mean_acc_0.9308.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV5/model.0107-loss_0.447-acc_0.888-val_loss_0.3523-val_acc_0.8881-val_fmeasure_0.8710-val_mcc_0.7722-val_mean_acc_0.8861.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV6/model.0125-loss_0.428-acc_0.897-val_loss_0.2631-val_acc_0.9252-val_fmeasure_0.9272-val_mcc_0.8544-val_mean_acc_0.9283.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV7/model.0196-loss_0.345-acc_0.935-val_loss_0.2863-val_acc_0.9195-val_fmeasure_0.9178-val_mcc_0.8392-val_mean_acc_0.9194.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV8/model.0136-loss_0.429-acc_0.907-val_loss_0.2853-val_acc_0.9375-val_fmeasure_0.9322-val_mcc_0.8761-val_mean_acc_0.9359.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV9/model.0153-loss_0.410-acc_0.913-val_loss_0.2296-val_acc_0.9265-val_fmeasure_0.9242-val_mcc_0.8528-val_mean_acc_0.9264.h5', '/home/mhubrich/checkpoints/adni/deepROI3_AD_1_CV10/model.0387-loss_0.256-acc_0.967-val_loss_0.2721-val_acc_0.9375-val_fmeasure_0.9383-val_mcc_0.8762-val_mean_acc_0.9378.h5']
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
                       [callbacks.early_stop(patience=45, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
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

