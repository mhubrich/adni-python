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
from cnn.keras.meanROI1.model import build_model
from cnn.keras.meanROI1.image_processing import inputs
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys
#sys.stdout = sys.stderr = open('output_first_7', 'w')

fold = str(sys.argv[1])
#pre = str(sys.argv[2])

# Training specific parameters
target_size = (18, 18, 18)
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = False
num_epoch = 750
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_meanROI1_1'
path_checkpoints = '/home/mhubrich/checkpoints/adni/meanROI1_pretrained_82_2_rui_final_CV' + fold
path_weights = None

def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + fold + '_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + fold + '_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED, 'binary')
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'predict', SEED, 'binary')

    # Set up the model
    if path_weights is None:
        model = build_model(1)
        sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        model = load_model(path_weights)

    model.load_weights('/home/mhubrich/checkpoints/adni/meanROI1_pretrained_82_2_rui/model.0048-loss_0.366-acc_0.976-val_loss_0.4185-val_acc_0.9213-val_fmeasure_0.9067-val_mcc_0.8389-val_mean_acc_0.9180.h5', by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(val_inputs,
                       [callbacks.early_stop(patience=100, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
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

