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
from cnn.keras.models.deepROI3.model_NC import build_model
from cnn.keras.deepROI3.image_processing import inputs
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])
#sys.stdout = sys.stderr = open('output_1_' + fold, 'w')

# Training specific parameters
target_size = (17, 17, 17)
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = True
num_epoch = 1000
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_deepROI3_NC'
path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI3_NC_1_CV' + fold
path_weights = None

#mod = ['/home/mhubrich/checkpoints/adni/AVG444_1_CV1/model.0357-loss_0.309-acc_0.966-val_loss_0.4842-val_acc_0.8794-val_fmeasure_0.8794-val_mcc_0.7590-val_mean_acc_0.8795.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV2/model.0159-loss_0.550-acc_0.878-val_loss_0.3468-val_acc_0.8889-val_fmeasure_0.8931-val_mcc_0.7781-val_mean_acc_0.8885.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV3/model.0440-loss_0.488-acc_0.901-val_loss_0.2656-val_acc_0.9275-val_fmeasure_0.9265-val_mcc_0.8566-val_mean_acc_0.9279.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV4/model.0297-loss_0.395-acc_0.936-val_loss_0.2155-val_acc_0.9434-val_fmeasure_0.9441-val_mcc_0.8886-val_mean_acc_0.9447.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV5/model.0241-loss_0.433-acc_0.922-val_loss_0.3891-val_acc_0.8811-val_fmeasure_0.8661-val_mcc_0.7600-val_mean_acc_0.8782.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV6/model.0184-loss_0.486-acc_0.904-val_loss_0.2749-val_acc_0.9184-val_fmeasure_0.9189-val_mcc_0.8381-val_mean_acc_0.9192.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV7/model.0232-loss_0.454-acc_0.913-val_loss_0.1351-val_acc_0.9664-val_fmeasure_0.9660-val_mcc_0.9337-val_mean_acc_0.9666.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV8/model.0187-loss_0.508-acc_0.902-val_loss_0.2162-val_acc_0.9375-val_fmeasure_0.9273-val_mcc_0.8744-val_mean_acc_0.9411.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV9/model.0274-loss_0.431-acc_0.924-val_loss_0.2381-val_acc_0.9485-val_fmeasure_0.9489-val_mcc_0.8996-val_mean_acc_0.9501.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV10/model.0356-loss_0.382-acc_0.937-val_loss_0.1543-val_acc_0.9688-val_fmeasure_0.9689-val_mcc_0.9394-val_mean_acc_0.9695.h5']

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

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(val_inputs,
                       [callbacks.early_stop(patience=80, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
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

