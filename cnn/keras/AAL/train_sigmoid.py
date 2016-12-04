from keras.models import load_model
from cnn.keras import callbacks
from keras.optimizers import SGD
from cnn.keras.AAL.model import build_model
from cnn.keras.AAL.image_processing import inputs
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys
#sys.stdout = sys.stderr = open('output_first_7', 'w')

fold = str(sys.argv[1])

# Training specific parameters
target_size = (18, 18, 18)
SEED = 0  # To deactivate seed, set to None
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = False
num_epoch = 500
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_AAL64'
path_checkpoints = '/home/mhubrich/checkpoints/adni/hinge_test_CV' + fold
path_weights = None


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED, 'binary')
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'val', SEED, 'binary')

    # Set up the model
    if path_weights is None:
        model = build_model(1)
        sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9, nesterov=True)
        model.compile(loss='squared_hinge', optimizer=sgd, metrics=['accuracy', 'fmeasure', 'binary_crossentropy'])
    else:
        model = load_model(path_weights)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL64_CV_10/model.0150-loss_0.468-acc_0.819-val_loss_0.3542-val_acc_0.8852.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL65_CV_10/model.0238-loss_0.430-acc_0.862-val_loss_0.4031-val_acc_0.8361.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL34_CV_10/model.0243-loss_0.333-acc_0.885-val_loss_0.2163-val_acc_0.9016.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL35_CV_10/model.0258-loss_0.411-acc_0.842-val_loss_0.2949-val_acc_0.9426.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL61_CV_10/model.0090-loss_0.443-acc_0.851-val_loss_0.2905-val_acc_0.9016.h5', by_name=True)
    #model.load_weights('/home/mhubrich/checkpoints/adni/AAL_diff_CV_10/model.34193-loss_0.478-acc_0.762-val_loss_0.3730-val_acc_0.9180.h5', by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            callbacks.early_stop(patience=100),
            callbacks.save_model(path_checkpoints, max_files=3, monitor=['val_loss', 'val_acc'])]

    g, _ = sort_groups(scans_train)

    # Start training
    hist = model.fit_generator(
        train_inputs,
        samples_per_epoch=train_inputs.nb_sample,
        nb_epoch=num_epoch,
        validation_data=val_inputs,
        nb_val_samples=val_inputs.nb_sample,
        callbacks=cbks,
        class_weight={0:max(len(g['Normal']), len(g['AD']))/float(len(g['Normal'])),
                      1:max(len(g['Normal']), len(g['AD']))/float(len(g['AD']))},
        verbose=2,
        max_q_size=32,
        nb_worker=1,
        pickle_safe=True)


if __name__ == "__main__":
    train()

