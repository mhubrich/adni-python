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
num_epoch = 400
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_meanROI1_1'
path_checkpoints = '/home/mhubrich/checkpoints/adni/meanROI1_acc_CV' + fold
path_weights = None

mod1 = ['/home/mhubrich/checkpoints/adni/meanROI1_1_CV1/model.0021-loss_0.972-acc_0.633-fmeasure_0.633-val_loss_0.6603-val_acc_0.7833-val_fmeasure_0.7833.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV2/model.0059-loss_0.957-acc_0.687-fmeasure_0.687-val_loss_0.6223-val_acc_0.7925-val_fmeasure_0.7925.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV3/model.0545-loss_0.190-acc_0.965-fmeasure_0.965-val_loss_0.9388-val_acc_0.8115-val_fmeasure_0.8115.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV4/model.0155-loss_0.557-acc_0.828-fmeasure_0.828-val_loss_0.4637-val_acc_0.8036-val_fmeasure_0.8036.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV5/model.0143-loss_0.597-acc_0.833-fmeasure_0.833-val_loss_0.2683-val_acc_0.8960-val_fmeasure_0.8960.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV6/model.0161-loss_0.299-acc_0.942-fmeasure_0.942-val_loss_0.4240-val_acc_0.9000-val_fmeasure_0.9000.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV7/model.0018-loss_0.984-acc_0.643-fmeasure_0.643-val_loss_0.6692-val_acc_0.8321-val_fmeasure_0.8321.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV8/model.0083-loss_0.650-acc_0.826-fmeasure_0.826-val_loss_0.5168-val_acc_0.8235-val_fmeasure_0.8235.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV9/model.0084-loss_0.905-acc_0.707-fmeasure_0.707-val_loss_0.5631-val_acc_0.7687-val_fmeasure_0.7687.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_CV10/model.0042-loss_0.983-acc_0.689-fmeasure_0.689-val_loss_0.6620-val_acc_0.7787-val_fmeasure_0.7787.h5']
mod2 = ['/home/mhubrich/checkpoints/adni/meanROI1_2_CV1/model.0065-loss_0.633-acc_0.823-fmeasure_0.823-val_loss_0.3696-val_acc_0.8750-val_fmeasure_0.8750.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV2/model.0074-loss_0.523-acc_0.863-fmeasure_0.863-val_loss_0.4597-val_acc_0.8208-val_fmeasure_0.8208.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV3/model.0185-loss_0.522-acc_0.863-fmeasure_0.863-val_loss_0.4270-val_acc_0.8361-val_fmeasure_0.8361.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV4/model.0116-loss_0.341-acc_0.924-fmeasure_0.924-val_loss_0.4180-val_acc_0.8839-val_fmeasure_0.8839.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV5/model.0051-loss_0.666-acc_0.813-fmeasure_0.813-val_loss_0.2977-val_acc_0.9440-val_fmeasure_0.9440.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV6/model.0146-loss_0.196-acc_0.971-fmeasure_0.971-val_loss_0.4990-val_acc_0.9000-val_fmeasure_0.9000.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV7/model.0049-loss_0.734-acc_0.773-fmeasure_0.773-val_loss_0.4241-val_acc_0.8473-val_fmeasure_0.8473.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV8/model.0086-loss_0.525-acc_0.879-fmeasure_0.879-val_loss_0.4480-val_acc_0.8235-val_fmeasure_0.8235.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV9/model.0028-loss_0.811-acc_0.748-fmeasure_0.748-val_loss_0.4598-val_acc_0.8209-val_fmeasure_0.8209.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_CV10/model.0158-loss_0.175-acc_0.977-fmeasure_0.977-val_loss_0.3901-val_acc_0.8770-val_fmeasure_0.8770.h5']
mod3 = ['/home/mhubrich/checkpoints/adni/meanROI1_3_CV1/model.0099-loss_0.336-acc_0.938-fmeasure_0.938-val_loss_0.2401-val_acc_0.9583-val_fmeasure_0.9583.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV2/model.0056-loss_0.487-acc_0.879-fmeasure_0.879-val_loss_0.2134-val_acc_0.9528-val_fmeasure_0.9528.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV3/model.0091-loss_0.549-acc_0.852-fmeasure_0.852-val_loss_0.3589-val_acc_0.8770-val_fmeasure_0.8770.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV4/model.0049-loss_0.533-acc_0.858-fmeasure_0.858-val_loss_0.4026-val_acc_0.8661-val_fmeasure_0.8661.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV5/model.0135-loss_0.273-acc_0.955-fmeasure_0.955-val_loss_0.1689-val_acc_0.9520-val_fmeasure_0.9520.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV6/model.0157-loss_0.244-acc_0.964-fmeasure_0.964-val_loss_0.1891-val_acc_0.9385-val_fmeasure_0.9385.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV7/model.0102-loss_0.349-acc_0.934-fmeasure_0.934-val_loss_0.4341-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV8/model.0121-loss_0.304-acc_0.942-fmeasure_0.942-val_loss_0.2499-val_acc_0.9265-val_fmeasure_0.9265.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV9/model.0129-loss_0.355-acc_0.938-fmeasure_0.938-val_loss_0.3543-val_acc_0.8881-val_fmeasure_0.8881.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_CV10/model.0057-loss_0.457-acc_0.902-fmeasure_0.902-val_loss_0.2062-val_acc_0.9590-val_fmeasure_0.9590.h5']
mod4 = ['/home/mhubrich/checkpoints/adni/meanROI1_4_CV1/model.0142-loss_0.600-acc_0.836-fmeasure_0.836-val_loss_0.3914-val_acc_0.8750-val_fmeasure_0.8750.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV2/model.0098-loss_0.611-acc_0.832-fmeasure_0.832-val_loss_0.3693-val_acc_0.8868-val_fmeasure_0.8868.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV3/model.0010-loss_0.974-acc_0.579-fmeasure_0.579-val_loss_0.6587-val_acc_0.7705-val_fmeasure_0.7705.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV4/model.0705-loss_0.572-acc_0.842-fmeasure_0.842-val_loss_0.6151-val_acc_0.7857-val_fmeasure_0.7857.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV5/model.0116-loss_0.655-acc_0.812-fmeasure_0.812-val_loss_0.3501-val_acc_0.8800-val_fmeasure_0.8800.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV6/model.0086-loss_0.586-acc_0.839-fmeasure_0.839-val_loss_0.2889-val_acc_0.9077-val_fmeasure_0.9077.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV7/model.0055-loss_0.655-acc_0.830-fmeasure_0.830-val_loss_0.4091-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV8/model.0566-loss_1.003-acc_0.699-fmeasure_0.699-val_loss_0.6878-val_acc_0.8235-val_fmeasure_0.8235.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV9/model.0179-loss_0.168-acc_0.978-fmeasure_0.978-val_loss_0.6024-val_acc_0.8657-val_fmeasure_0.8657.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_CV10/model.0032-loss_0.754-acc_0.786-fmeasure_0.786-val_loss_0.3872-val_acc_0.8689-val_fmeasure_0.8689.h5']
mod5 = ['/home/mhubrich/checkpoints/adni/meanROI1_5_CV1/model.0275-loss_0.699-acc_0.778-fmeasure_0.778-val_loss_0.5027-val_acc_0.8250-val_fmeasure_0.8250.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV2/model.0266-loss_0.870-acc_0.683-fmeasure_0.683-val_loss_0.5556-val_acc_0.8774-val_fmeasure_0.8774.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV3/model.0039-loss_0.839-acc_0.708-fmeasure_0.708-val_loss_0.5201-val_acc_0.7869-val_fmeasure_0.7869.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV4/model.0619-loss_0.190-acc_0.953-fmeasure_0.953-val_loss_1.1813-val_acc_0.7946-val_fmeasure_0.7946.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV5/model.0105-loss_0.888-acc_0.705-fmeasure_0.705-val_loss_0.6110-val_acc_0.8400-val_fmeasure_0.8400.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV6/model.0015-loss_0.960-acc_0.542-fmeasure_0.542-val_loss_0.6891-val_acc_0.8538-val_fmeasure_0.8538.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV7/model.0093-loss_0.766-acc_0.743-fmeasure_0.743-val_loss_0.4344-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV8/model.0160-loss_0.773-acc_0.731-fmeasure_0.731-val_loss_0.4456-val_acc_0.9044-val_fmeasure_0.9044.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV9/model.0020-loss_0.962-acc_0.530-fmeasure_0.530-val_loss_0.6862-val_acc_0.8134-val_fmeasure_0.8134.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_CV10/model.0015-loss_0.949-acc_0.601-fmeasure_0.601-val_loss_0.6812-val_acc_0.8689-val_fmeasure_0.8689.h5']
mod_diff = ['/home/mhubrich/checkpoints/adni/meanROI1_diff_CV1/model.0937-loss_0.689-acc_0.774-fmeasure_0.774-val_loss_0.4614-val_acc_0.8417-val_fmeasure_0.8417.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV2/model.5037-loss_0.706-acc_0.718-fmeasure_0.718-val_loss_0.4608-val_acc_0.8774-val_fmeasure_0.8774.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV3/model.2601-loss_0.658-acc_0.795-fmeasure_0.795-val_loss_0.5038-val_acc_0.7869-val_fmeasure_0.7869.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV4/model.0613-loss_0.792-acc_0.672-fmeasure_0.672-val_loss_0.5606-val_acc_0.7857-val_fmeasure_0.7857.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV5/model.2139-loss_0.669-acc_0.773-fmeasure_0.773-val_loss_0.4010-val_acc_0.8880-val_fmeasure_0.8880.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV6/model.4138-loss_0.695-acc_0.775-fmeasure_0.775-val_loss_0.4453-val_acc_0.8615-val_fmeasure_0.8615.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV7/model.3922-loss_0.680-acc_0.774-fmeasure_0.774-val_loss_0.4568-val_acc_0.8702-val_fmeasure_0.8702.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV8/model.1426-loss_0.695-acc_0.760-fmeasure_0.760-val_loss_0.4499-val_acc_0.8456-val_fmeasure_0.8456.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV9/model.3036-loss_0.655-acc_0.777-fmeasure_0.777-val_loss_0.4477-val_acc_0.8507-val_fmeasure_0.8507.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_CV10/model.0585-loss_0.701-acc_0.780-fmeasure_0.780-val_loss_0.3833-val_acc_0.9016-val_fmeasure_0.9016.h5']


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

