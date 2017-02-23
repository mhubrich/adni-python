##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

import nibabel as nib
from cnn.keras import callbacks
from cnn.keras.evaluation_callback2 import Evaluation
from keras.optimizers import SGD
from cnn.keras.final_metaROI.model import build_model
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])

# Training specific parameters
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
num_epoch = 2000
# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC'
path_checkpoints = '/home/mhubrich/checkpoints/adni/final_metaROI_merged_5_2_pSMC_CV' + fold
path_model = [['/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV1/model.0680-loss_0.581-acc_0.813-val_loss_0.4219-val_acc_0.8529-val_mean_acc_0.8495.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV2/model.0454-loss_0.593-acc_0.808-val_loss_0.4248-val_acc_0.8467-val_mean_acc_0.8464.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV3/model.1419-loss_0.733-acc_0.790-val_loss_0.4849-val_acc_0.8577-val_mean_acc_0.8510.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV4/model.0337-loss_0.871-acc_0.545-val_loss_0.6854-val_acc_0.9071-val_mean_acc_0.9063.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV5/model.0863-loss_0.575-acc_0.811-val_loss_0.4495-val_acc_0.8498-val_mean_acc_0.8495.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV6/model.0503-loss_0.588-acc_0.818-val_loss_0.3446-val_acc_0.8855-val_mean_acc_0.8834.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV7/model.1763-loss_0.555-acc_0.835-val_loss_0.3257-val_acc_0.8945-val_mean_acc_0.8938.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV8/model.1582-loss_0.569-acc_0.834-val_loss_0.3937-val_acc_0.8504-val_mean_acc_0.8483.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV9/model.0728-loss_0.618-acc_0.793-val_loss_0.3833-val_acc_0.8745-val_mean_acc_0.8688.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_pSMC_CV10/model.1163-loss_0.581-acc_0.826-val_loss_0.3986-val_acc_0.8476-val_mean_acc_0.8475.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV1/model.0385-loss_0.704-acc_0.741-val_loss_0.5256-val_acc_0.7794-val_mean_acc_0.7639.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV2/model.0783-loss_0.651-acc_0.752-val_loss_0.5088-val_acc_0.7778-val_mean_acc_0.7773.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV3/model.2496-loss_0.684-acc_0.763-val_loss_0.5431-val_acc_0.7903-val_mean_acc_0.7869.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV4/model.1062-loss_0.713-acc_0.716-val_loss_0.4541-val_acc_0.8357-val_mean_acc_0.8323.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV5/model.0150-loss_0.819-acc_0.646-val_loss_0.6572-val_acc_0.7729-val_mean_acc_0.7454.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV6/model.0615-loss_0.659-acc_0.758-val_loss_0.5080-val_acc_0.7863-val_mean_acc_0.7810.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV7/model.0615-loss_0.704-acc_0.735-val_loss_0.4903-val_acc_0.7891-val_mean_acc_0.7877.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV8/model.1332-loss_0.678-acc_0.757-val_loss_0.4799-val_acc_0.7953-val_mean_acc_0.7906.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV9/model.1180-loss_0.678-acc_0.739-val_loss_0.5124-val_acc_0.7934-val_mean_acc_0.7861.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_pSMC_CV10/model.0207-loss_0.773-acc_0.718-val_loss_0.5695-val_acc_0.7918-val_mean_acc_0.7911.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV1/model.0334-loss_0.623-acc_0.837-val_loss_0.3320-val_acc_0.9007-val_mean_acc_0.8995.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV2/model.1021-loss_0.519-acc_0.884-val_loss_0.3400-val_acc_0.8812-val_mean_acc_0.8810.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV3/model.1397-loss_0.861-acc_0.662-val_loss_0.4523-val_acc_0.8951-val_mean_acc_0.8881.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV4/model.0260-loss_0.761-acc_0.778-val_loss_0.4046-val_acc_0.8893-val_mean_acc_0.8858.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV5/model.0274-loss_0.612-acc_0.842-val_loss_0.3463-val_acc_0.8938-val_mean_acc_0.8953.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV6/model.0632-loss_0.610-acc_0.852-val_loss_0.3095-val_acc_0.9084-val_mean_acc_0.9074.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV7/model.1137-loss_0.569-acc_0.865-val_loss_0.2573-val_acc_0.9219-val_mean_acc_0.9216.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV8/model.0334-loss_0.623-acc_0.841-val_loss_0.3541-val_acc_0.8819-val_mean_acc_0.8817.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV9/model.1211-loss_0.566-acc_0.860-val_loss_0.2818-val_acc_0.9077-val_mean_acc_0.9072.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_pSMC_CV10/model.1422-loss_0.512-acc_0.888-val_loss_0.3126-val_acc_0.8922-val_mean_acc_0.8922.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV1/model.0676-loss_0.623-acc_0.834-val_loss_0.4130-val_acc_0.8934-val_mean_acc_0.8885.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV2/model.0110-loss_0.757-acc_0.776-val_loss_0.4865-val_acc_0.8429-val_mean_acc_0.8425.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV3/model.0402-loss_0.953-acc_0.693-val_loss_0.6811-val_acc_0.8015-val_mean_acc_0.8053.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV4/model.1668-loss_0.702-acc_0.803-val_loss_0.3367-val_acc_0.9036-val_mean_acc_0.9034.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV5/model.0236-loss_0.702-acc_0.796-val_loss_0.4153-val_acc_0.8791-val_mean_acc_0.8794.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV6/model.0436-loss_0.646-acc_0.821-val_loss_0.4452-val_acc_0.8550-val_mean_acc_0.8548.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV7/model.0580-loss_0.668-acc_0.818-val_loss_0.4254-val_acc_0.8555-val_mean_acc_0.8557.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV8/model.0428-loss_0.679-acc_0.816-val_loss_0.3635-val_acc_0.8937-val_mean_acc_0.8931.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV9/model.0076-loss_0.806-acc_0.775-val_loss_0.5168-val_acc_0.8450-val_mean_acc_0.8472.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_pSMC_CV10/model.0846-loss_0.648-acc_0.829-val_loss_0.3473-val_acc_0.8922-val_mean_acc_0.8921.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV1/model.3299-loss_0.691-acc_0.720-val_loss_0.5234-val_acc_0.7721-val_mean_acc_0.7565.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV2/model.0167-loss_0.826-acc_0.556-val_loss_0.6772-val_acc_0.7395-val_mean_acc_0.7389.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV3/model.0309-loss_0.801-acc_0.594-val_loss_0.6588-val_acc_0.7341-val_mean_acc_0.7338.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV4/model.1166-loss_0.729-acc_0.688-val_loss_0.4959-val_acc_0.8500-val_mean_acc_0.8485.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV5/model.0394-loss_0.760-acc_0.646-val_loss_0.6265-val_acc_0.7802-val_mean_acc_0.7700.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV6/model.0417-loss_0.739-acc_0.715-val_loss_0.5904-val_acc_0.7672-val_mean_acc_0.7621.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV7/model.1408-loss_0.710-acc_0.700-val_loss_0.5195-val_acc_0.8164-val_mean_acc_0.8161.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV8/model.3737-loss_0.700-acc_0.718-val_loss_0.5024-val_acc_0.8189-val_mean_acc_0.8143.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV9/model.0262-loss_0.801-acc_0.591-val_loss_0.6629-val_acc_0.7897-val_mean_acc_0.7872.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_pSMC_CV10/model.0202-loss_0.821-acc_0.588-val_loss_0.6741-val_acc_0.7472-val_mean_acc_0.7466.h5']]

ranges = [(range(65, 73), range(28, 36), range(51, 59)),
          (range(73, 79), range(39, 45), range(27, 33)),
          (range(39, 49), range(37, 47), range(46, 56)),
          (range(13, 26), range(28, 41), range(47, 60)),
          (range(11, 16), range(35, 40), range(31, 36))]

def load_data(scans, flip=False):
    groups, _ = sort_groups(scans)
    nb_samples = 0
    for c in classes:
        assert groups[c] is not None, \
            'Could not find class %s' % c
        nb_samples += len(groups[c])
    if flip:
        nb_samples *= 2
    y = np.zeros(nb_samples, dtype=np.int32)
    X1 = np.zeros((nb_samples, 1, 8, 8, 8), dtype=np.float32)
    X2 = np.zeros((nb_samples, 1, 6, 6, 6), dtype=np.float32)
    X3 = np.zeros((nb_samples, 1, 10, 10, 10), dtype=np.float32)
    X4 = np.zeros((nb_samples, 1, 13, 13, 13), dtype=np.float32)
    X5 = np.zeros((nb_samples, 1, 5, 5, 5), dtype=np.float32)
    X = [X1, X2, X3, X4, X5]
    i = 0
    for c in classes:
        for scan in groups[c]:
            s = nib.load(scan.path).get_data()
            y[i] = 0 if scan.group == classes[0] else 1
            for j in range(len(X)):
                X[j][i] = s[ranges[j][0],:,:][:,ranges[j][1],:][:,:,ranges[j][2]]
            i += 1
            if flip:
                for j in range(len(X)):
                    X[j][i] = np.flipud(s)[ranges[j][0],:,:][:,ranges[j][1],:][:,:,ranges[j][2]]
                y[i] = y[i-1]
                i += 1
            del s
    return X, y


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean3/' + fold + '_train', 'nii')
    x_train, y_train = load_data(scans_train, flip=True)

    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean3/' + fold + '_val', 'nii')
    x_val, y_val = load_data(scans_val, flip=False)

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    for i in range(5):
        model.load_weights(path_model[i][int(fold)-1], by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            Evaluation(x_val, y_val, batch_size,
                       [callbacks.early_stop(patience=500, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=1, monitor=['val_loss', 'val_acc', 'val_mean_acc'])])]

    g, _ = sort_groups(scans_train)

    hist = model.fit(x=x_train,
                     y=y_train,
                     nb_epoch=num_epoch,
                     callbacks=cbks,
                     class_weight={0:max(len(g['Normal']), len(g['AD']))/float(len(g['Normal'])),
                                   1:max(len(g['Normal']), len(g['AD']))/float(len(g['AD']))},
                     batch_size=batch_size,
                     shuffle=True,
                     verbose=2)


if __name__ == "__main__":
    train()

