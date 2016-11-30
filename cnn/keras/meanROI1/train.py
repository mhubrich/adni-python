from keras.models import load_model
from cnn.keras import callbacks
from keras.metrics import fmeasure
from keras.optimizers import SGD
from cnn.keras.meanROI1.model import build_model
from cnn.keras.meanROI1.image_processing import inputs
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
path_ADNI = '/home/mhubrich/ADNI_intnorm_meanROI1_1'
path_checkpoints = '/home/mhubrich/checkpoints/adni/meanROI1_F1_CV' + fold
path_weights = None

mod1 = ['/home/mhubrich/checkpoints/adni/meanROI1_1_CV1/model.0319-loss_0.457-acc_0.876-fmeasure_0.876-val_loss_0.5514-val_acc_0.7750-val_fmeasure_0.7750.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV2/model.0059-loss_0.957-acc_0.687-fmeasure_0.687-val_loss_0.6223-val_acc_0.7925-val_fmeasure_0.7925.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV3/model.0545-loss_0.190-acc_0.965-fmeasure_0.965-val_loss_0.9388-val_acc_0.8115-val_fmeasure_0.8115.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV4/model.0159-loss_0.571-acc_0.838-fmeasure_0.838-val_loss_0.4989-val_acc_0.8036-val_fmeasure_0.8036.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV5/model.0143-loss_0.597-acc_0.833-fmeasure_0.833-val_loss_0.2683-val_acc_0.8960-val_fmeasure_0.8960.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV6/model.0161-loss_0.299-acc_0.942-fmeasure_0.942-val_loss_0.4240-val_acc_0.9000-val_fmeasure_0.9000.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV7/model.0018-loss_0.984-acc_0.643-fmeasure_0.643-val_loss_0.6692-val_acc_0.8321-val_fmeasure_0.8321.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV8/model.0093-loss_0.589-acc_0.851-fmeasure_0.851-val_loss_0.5351-val_acc_0.8235-val_fmeasure_0.8235.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV9/model.0117-loss_0.704-acc_0.803-fmeasure_0.803-val_loss_0.5974-val_acc_0.7612-val_fmeasure_0.7612.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV10/model.0042-loss_0.983-acc_0.689-fmeasure_0.689-val_loss_0.6620-val_acc_0.7787-val_fmeasure_0.7787.h5']
mod2 = ['/home/mhubrich/checkpoints/adni/meanROI1_2_CV1/model.0061-loss_0.607-acc_0.819-fmeasure_0.819-val_loss_0.3713-val_acc_0.8750-val_fmeasure_0.8750.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV2/model.0141-loss_0.187-acc_0.975-fmeasure_0.975-val_loss_0.5404-val_acc_0.8208-val_fmeasure_0.8208.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV3/model.0165-loss_0.596-acc_0.843-fmeasure_0.843-val_loss_0.4432-val_acc_0.8279-val_fmeasure_0.8279.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV4/model.0116-loss_0.341-acc_0.924-fmeasure_0.924-val_loss_0.4180-val_acc_0.8839-val_fmeasure_0.8839.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV5/model.0051-loss_0.666-acc_0.813-fmeasure_0.813-val_loss_0.2977-val_acc_0.9440-val_fmeasure_0.9440.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV6/model.0146-loss_0.196-acc_0.971-fmeasure_0.971-val_loss_0.4990-val_acc_0.9000-val_fmeasure_0.9000.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV7/model.0159-loss_0.146-acc_0.982-fmeasure_0.982-val_loss_0.6130-val_acc_0.8473-val_fmeasure_0.8473.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV8/model.0118-loss_0.303-acc_0.942-fmeasure_0.942-val_loss_0.5196-val_acc_0.8162-val_fmeasure_0.8162.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV9/model.0039-loss_0.789-acc_0.764-fmeasure_0.764-val_loss_0.5178-val_acc_0.8209-val_fmeasure_0.8209.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV10/model.0161-loss_0.188-acc_0.972-fmeasure_0.972-val_loss_0.4183-val_acc_0.8770-val_fmeasure_0.8770.h5']
mod3 = ['/home/mhubrich/checkpoints/adni/meanROI1_3_CV1/model.0099-loss_0.336-acc_0.938-fmeasure_0.938-val_loss_0.2401-val_acc_0.9583-val_fmeasure_0.9583.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV2/model.0046-loss_0.553-acc_0.852-fmeasure_0.852-val_loss_0.2538-val_acc_0.9528-val_fmeasure_0.9528.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV3/model.0112-loss_0.420-acc_0.919-fmeasure_0.919-val_loss_0.3999-val_acc_0.8770-val_fmeasure_0.8770.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV4/model.0091-loss_0.393-acc_0.919-fmeasure_0.919-val_loss_0.4964-val_acc_0.8571-val_fmeasure_0.8571.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV5/model.0135-loss_0.273-acc_0.955-fmeasure_0.955-val_loss_0.1689-val_acc_0.9520-val_fmeasure_0.9520.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV6/model.0106-loss_0.382-acc_0.923-fmeasure_0.923-val_loss_0.1939-val_acc_0.9385-val_fmeasure_0.9385.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV7/model.0102-loss_0.349-acc_0.934-fmeasure_0.934-val_loss_0.4341-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV8/model.0121-loss_0.304-acc_0.942-fmeasure_0.942-val_loss_0.2499-val_acc_0.9265-val_fmeasure_0.9265.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV9/model.0129-loss_0.355-acc_0.938-fmeasure_0.938-val_loss_0.3543-val_acc_0.8881-val_fmeasure_0.8881.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV10/model.0031-loss_0.562-acc_0.861-fmeasure_0.861-val_loss_0.2125-val_acc_0.9590-val_fmeasure_0.9590.h5']
mod4 = ['/home/mhubrich/checkpoints/adni/meanROI1_4_CV1/model.0142-loss_0.600-acc_0.836-fmeasure_0.836-val_loss_0.3914-val_acc_0.8750-val_fmeasure_0.8750.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV2/model.0098-loss_0.611-acc_0.832-fmeasure_0.832-val_loss_0.3693-val_acc_0.8868-val_fmeasure_0.8868.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV3/model.0010-loss_0.974-acc_0.579-fmeasure_0.579-val_loss_0.6587-val_acc_0.7705-val_fmeasure_0.7705.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV4/model.0711-loss_0.543-acc_0.866-fmeasure_0.866-val_loss_0.7166-val_acc_0.7679-val_fmeasure_0.7679.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV5/model.0155-loss_0.478-acc_0.882-fmeasure_0.882-val_loss_0.3841-val_acc_0.8800-val_fmeasure_0.8800.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV6/model.0086-loss_0.586-acc_0.839-fmeasure_0.839-val_loss_0.2889-val_acc_0.9077-val_fmeasure_0.9077.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV7/model.0055-loss_0.655-acc_0.830-fmeasure_0.830-val_loss_0.4091-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV8/model.0330-loss_0.973-acc_0.690-fmeasure_0.690-val_loss_0.6482-val_acc_0.8162-val_fmeasure_0.8162.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV9/model.0179-loss_0.168-acc_0.978-fmeasure_0.978-val_loss_0.6024-val_acc_0.8657-val_fmeasure_0.8657.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV10/model.0031-loss_0.782-acc_0.752-fmeasure_0.752-val_loss_0.4141-val_acc_0.8689-val_fmeasure_0.8689.h5']
mod5 = ['/home/mhubrich/checkpoints/adni/meanROI1_5_CV1/model.0145-loss_0.762-acc_0.740-fmeasure_0.740-val_loss_0.5527-val_acc_0.8250-val_fmeasure_0.8250.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV2/model.0266-loss_0.870-acc_0.683-fmeasure_0.683-val_loss_0.5556-val_acc_0.8774-val_fmeasure_0.8774.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV3/model.0040-loss_0.833-acc_0.694-fmeasure_0.694-val_loss_0.5207-val_acc_0.7869-val_fmeasure_0.7869.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV4/model.0619-loss_0.190-acc_0.953-fmeasure_0.953-val_loss_1.1813-val_acc_0.7946-val_fmeasure_0.7946.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV5/model.0104-loss_0.903-acc_0.664-fmeasure_0.664-val_loss_0.6262-val_acc_0.8400-val_fmeasure_0.8400.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV6/model.0015-loss_0.960-acc_0.542-fmeasure_0.542-val_loss_0.6891-val_acc_0.8538-val_fmeasure_0.8538.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV7/model.0071-loss_0.765-acc_0.736-fmeasure_0.736-val_loss_0.4424-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV8/model.0160-loss_0.773-acc_0.731-fmeasure_0.731-val_loss_0.4456-val_acc_0.9044-val_fmeasure_0.9044.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV9/model.0020-loss_0.962-acc_0.530-fmeasure_0.530-val_loss_0.6862-val_acc_0.8134-val_fmeasure_0.8134.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV10/model.0015-loss_0.949-acc_0.601-fmeasure_0.601-val_loss_0.6812-val_acc_0.8689-val_fmeasure_0.8689.h5']
mod_diff = ['/home/mhubrich/checkpoints/adni/meanROI1_diff_CV1/model.0863-loss_0.695-acc_0.768-fmeasure_0.768-val_loss_0.4785-val_acc_0.8417-val_fmeasure_0.8417.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV2/model.5037-loss_0.706-acc_0.718-fmeasure_0.718-val_loss_0.4608-val_acc_0.8774-val_fmeasure_0.8774.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV3/model.1753-loss_0.678-acc_0.786-fmeasure_0.786-val_loss_0.5009-val_acc_0.7787-val_fmeasure_0.7787.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV4/model.1753-loss_0.713-acc_0.736-fmeasure_0.736-val_loss_0.5763-val_acc_0.7857-val_fmeasure_0.7857.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV5/model.2139-loss_0.669-acc_0.773-fmeasure_0.773-val_loss_0.4010-val_acc_0.8880-val_fmeasure_0.8880.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV6/model.0485-loss_0.717-acc_0.783-fmeasure_0.783-val_loss_0.4614-val_acc_0.8538-val_fmeasure_0.8538.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV7/model.3922-loss_0.680-acc_0.774-fmeasure_0.774-val_loss_0.4568-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV8/model.1426-loss_0.695-acc_0.760-fmeasure_0.760-val_loss_0.4499-val_acc_0.8456-val_fmeasure_0.8456.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV9/model.3036-loss_0.655-acc_0.777-fmeasure_0.777-val_loss_0.4477-val_acc_0.8507-val_fmeasure_0.8507.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV10/model.0583-loss_0.711-acc_0.780-fmeasure_0.780-val_loss_0.3915-val_acc_0.9016-val_fmeasure_0.9016.h5']


def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV/' + fold + '_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV/' + fold + '_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED)
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'val', SEED)

    # Set up the model
    if path_weights is None:
        model = build_model(2)
        sgd = SGD(lr=0.001, decay=0.000001, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', fmeasure])
    else:
        model = load_model(path_weights)
    model.load_weights(mod1[int(fold)-1], by_name=True)
    model.load_weights(mod2[int(fold)-1], by_name=True)
    model.load_weights(mod3[int(fold)-1], by_name=True)
    model.load_weights(mod4[int(fold)-1], by_name=True)
    model.load_weights(mod5[int(fold)-1], by_name=True)
    model.load_weights(mod_diff[int(fold)-1], by_name=True)

    # Define callbacks
    cbks = [callbacks.print_history(),
            callbacks.flush(),
            callbacks.early_stopping(max_acc=0.99, patience=10),
            callbacks.save_model(path_checkpoints, max_files=3)]

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

