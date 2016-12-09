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

# Training specific parameters
target_size = (18, 18, 18)
classes = ['Normal', 'AD']
batch_size = 32
load_all_scans = False
num_epoch = 750
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_meanROI1_1'
path_checkpoints = '/home/mhubrich/checkpoints/adni/meanROI1_pretrained_acc_CV' + fold
path_weights = None

mod_diff = ['/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV1/model.3999-loss_0.616-acc_0.764-val_loss_0.4931-val_acc_0.7660-val_fmeasure_0.7227-val_mcc_0.5563-val_mean_acc_0.7920.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV2/model.3650-loss_0.627-acc_0.756-val_loss_0.5086-val_acc_0.7386-val_fmeasure_0.7059-val_mcc_0.5153-val_mean_acc_0.7680.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV3/model.4393-loss_0.664-acc_0.722-val_loss_0.5294-val_acc_0.7609-val_fmeasure_0.7027-val_mcc_0.5409-val_mean_acc_0.7882.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV4/model.5698-loss_0.605-acc_0.762-val_loss_0.4502-val_acc_0.8239-val_fmeasure_0.7971-val_mcc_0.6636-val_mean_acc_0.8422.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV5/model.7423-loss_0.594-acc_0.754-val_loss_0.5027-val_acc_0.7692-val_fmeasure_0.6916-val_mcc_0.5314-val_mean_acc_0.7836.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV6/model.7119-loss_0.584-acc_0.773-val_loss_0.4354-val_acc_0.8163-val_fmeasure_0.7805-val_mcc_0.6581-val_mean_acc_0.8456.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV7/model.4357-loss_0.619-acc_0.754-val_loss_0.4856-val_acc_0.7785-val_fmeasure_0.7273-val_mcc_0.5810-val_mean_acc_0.8090.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV8/model.5878-loss_0.598-acc_0.772-val_loss_0.4702-val_acc_0.8125-val_fmeasure_0.7647-val_mcc_0.6242-val_mean_acc_0.8249.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV9/model.6151-loss_0.614-acc_0.759-val_loss_0.4039-val_acc_0.8456-val_fmeasure_0.8293-val_mcc_0.6959-val_mean_acc_0.8524.h5', '/home/mhubrich/checkpoints/adni/meanROI1_diff_pretrained_CV10/model.9240-loss_0.600-acc_0.773-val_loss_0.4786-val_acc_0.7438-val_fmeasure_0.7092-val_mcc_0.5181-val_mean_acc_0.7693.h5']
mod1 = ['/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV1/model.0207-loss_0.489-acc_0.847-val_loss_0.4284-val_acc_0.8582-val_fmeasure_0.8507-val_mcc_0.7187-val_mean_acc_0.8609.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV2/model.0172-loss_0.484-acc_0.861-val_loss_0.5194-val_acc_0.7778-val_fmeasure_0.7875-val_mcc_0.5549-val_mean_acc_0.7771.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV3/model.0502-loss_0.448-acc_0.865-val_loss_0.3300-val_acc_0.8913-val_fmeasure_0.8855-val_mcc_0.7821-val_mean_acc_0.8914.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV4/model.0259-loss_0.478-acc_0.850-val_loss_0.2789-val_acc_0.9119-val_fmeasure_0.9079-val_mcc_0.8247-val_mean_acc_0.9133.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV5/model.0072-loss_0.664-acc_0.758-val_loss_0.5731-val_acc_0.7063-val_fmeasure_0.5714-val_mcc_0.4029-val_mean_acc_0.7300.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV6/model.0666-loss_0.321-acc_0.921-val_loss_0.3401-val_acc_0.8844-val_fmeasure_0.8811-val_mcc_0.7686-val_mean_acc_0.8845.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV7/model.0224-loss_0.463-acc_0.858-val_loss_0.3315-val_acc_0.8926-val_fmeasure_0.8857-val_mcc_0.7857-val_mean_acc_0.8942.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV8/model.0268-loss_0.475-acc_0.858-val_loss_0.3401-val_acc_0.8828-val_fmeasure_0.8649-val_mcc_0.7624-val_mean_acc_0.8836.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV9/model.0256-loss_0.475-acc_0.856-val_loss_0.3723-val_acc_0.8750-val_fmeasure_0.8661-val_mcc_0.7513-val_mean_acc_0.8775.h5', '/home/mhubrich/checkpoints/adni/meanROI1_1_pretrained_CV10/model.0123-loss_0.574-acc_0.810-val_loss_0.4782-val_acc_0.7500-val_fmeasure_0.7436-val_mcc_0.5056-val_mean_acc_0.7536.h5']
mod2 = ['/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV1/model.0075-loss_0.609-acc_0.795-val_loss_0.5408-val_acc_0.7376-val_fmeasure_0.7132-val_mcc_0.4805-val_mean_acc_0.7435.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV2/model.0166-loss_0.531-acc_0.827-val_loss_0.4831-val_acc_0.7843-val_fmeasure_0.7724-val_mcc_0.5871-val_mean_acc_0.7971.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV3/model.0624-loss_0.484-acc_0.858-val_loss_0.3888-val_acc_0.8261-val_fmeasure_0.8033-val_mcc_0.6563-val_mean_acc_0.8338.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV4/model.0464-loss_0.388-acc_0.901-val_loss_0.3714-val_acc_0.8553-val_fmeasure_0.8435-val_mcc_0.7146-val_mean_acc_0.8604.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV5/model.0209-loss_0.503-acc_0.835-val_loss_0.4461-val_acc_0.7972-val_fmeasure_0.7434-val_mcc_0.5859-val_mean_acc_0.8031.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV6/model.0372-loss_0.402-acc_0.901-val_loss_0.4098-val_acc_0.8503-val_fmeasure_0.8406-val_mcc_0.7024-val_mean_acc_0.8530.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV7/model.0285-loss_0.438-acc_0.871-val_loss_0.4503-val_acc_0.8188-val_fmeasure_0.7874-val_mcc_0.6519-val_mean_acc_0.8375.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV8/model.0268-loss_0.446-acc_0.884-val_loss_0.4496-val_acc_0.8047-val_fmeasure_0.7475-val_mcc_0.6126-val_mean_acc_0.8242.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV9/model.0212-loss_0.499-acc_0.848-val_loss_0.4130-val_acc_0.8235-val_fmeasure_0.8095-val_mcc_0.6484-val_mean_acc_0.8263.h5', '/home/mhubrich/checkpoints/adni/meanROI1_2_pretrained_CV10/model.0247-loss_0.482-acc_0.857-val_loss_0.3525-val_acc_0.8500-val_fmeasure_0.8442-val_mcc_0.7092-val_mean_acc_0.8566.h5']
mod3 = ['/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV1/model.0071-loss_0.408-acc_0.900-val_loss_0.3851-val_acc_0.8582-val_fmeasure_0.8551-val_mcc_0.7165-val_mean_acc_0.8585.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV2/model.0125-loss_0.340-acc_0.928-val_loss_0.3616-val_acc_0.8627-val_fmeasure_0.8645-val_mcc_0.7291-val_mean_acc_0.8641.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV3/model.0158-loss_0.451-acc_0.878-val_loss_0.2923-val_acc_0.8841-val_fmeasure_0.8788-val_mcc_0.7677-val_mean_acc_0.8838.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV4/model.0078-loss_0.412-acc_0.890-val_loss_0.2652-val_acc_0.8931-val_fmeasure_0.8889-val_mcc_0.7865-val_mean_acc_0.8938.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV5/model.0052-loss_0.504-acc_0.844-val_loss_0.3726-val_acc_0.8741-val_fmeasure_0.8548-val_mcc_0.7437-val_mean_acc_0.8719.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV6/model.0144-loss_0.285-acc_0.941-val_loss_0.3277-val_acc_0.8776-val_fmeasure_0.8750-val_mcc_0.7550-val_mean_acc_0.8775.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV7/model.0039-loss_0.475-acc_0.862-val_loss_0.2453-val_acc_0.9195-val_fmeasure_0.9155-val_mcc_0.8389-val_mean_acc_0.9200.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV8/model.0065-loss_0.411-acc_0.886-val_loss_0.2306-val_acc_0.9219-val_fmeasure_0.9123-val_mcc_0.8419-val_mean_acc_0.9209.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV9/model.0076-loss_0.390-acc_0.891-val_loss_0.2606-val_acc_0.8971-val_fmeasure_0.8939-val_mcc_0.7939-val_mean_acc_0.8970.h5', '/home/mhubrich/checkpoints/adni/meanROI1_3_pretrained_CV10/model.0172-loss_0.295-acc_0.943-val_loss_0.2714-val_acc_0.9250-val_fmeasure_0.9250-val_mcc_0.8526-val_mean_acc_0.9263.h5']
mod4 = ['/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV1/model.0102-loss_0.648-acc_0.771-val_loss_0.5129-val_acc_0.7234-val_fmeasure_0.6977-val_mcc_0.4517-val_mean_acc_0.7289.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV2/model.0422-loss_0.488-acc_0.855-val_loss_0.4692-val_acc_0.8039-val_fmeasure_0.8101-val_mcc_0.6085-val_mean_acc_0.8037.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV3/model.0060-loss_0.721-acc_0.736-val_loss_0.5128-val_acc_0.7536-val_fmeasure_0.7119-val_mcc_0.5128-val_mean_acc_0.7643.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV4/model.0254-loss_0.507-acc_0.846-val_loss_0.3608-val_acc_0.8365-val_fmeasure_0.8267-val_mcc_0.6743-val_mean_acc_0.8386.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV5/model.0255-loss_0.504-acc_0.843-val_loss_0.4256-val_acc_0.8112-val_fmeasure_0.7731-val_mcc_0.6135-val_mean_acc_0.8105.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV6/model.0248-loss_0.518-acc_0.848-val_loss_0.3527-val_acc_0.8435-val_fmeasure_0.8345-val_mcc_0.6881-val_mean_acc_0.8453.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV7/model.0410-loss_0.483-acc_0.868-val_loss_0.3814-val_acc_0.8188-val_fmeasure_0.8000-val_mcc_0.6404-val_mean_acc_0.8239.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV8/model.0292-loss_0.493-acc_0.859-val_loss_0.4311-val_acc_0.7969-val_fmeasure_0.7547-val_mcc_0.5879-val_mean_acc_0.8006.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV9/model.0230-loss_0.527-acc_0.835-val_loss_0.3278-val_acc_0.8529-val_fmeasure_0.8462-val_mcc_0.7057-val_mean_acc_0.8533.h5', '/home/mhubrich/checkpoints/adni/meanROI1_4_pretrained_CV10/model.0412-loss_0.490-acc_0.860-val_loss_0.4018-val_acc_0.8562-val_fmeasure_0.8589-val_mcc_0.7130-val_mean_acc_0.8562.h5']
mod5 = ['/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV1/model.0130-loss_0.686-acc_0.724-val_loss_0.5716-val_acc_0.7021-val_fmeasure_0.7000-val_mcc_0.4042-val_mean_acc_0.7021.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV2/model.0454-loss_0.590-acc_0.774-val_loss_0.5238-val_acc_0.6993-val_fmeasure_0.6761-val_mcc_0.4200-val_mean_acc_0.7141.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV3/model.0151-loss_0.715-acc_0.692-val_loss_0.5576-val_acc_0.7681-val_fmeasure_0.7288-val_mcc_0.5427-val_mean_acc_0.7797.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV4/model.0555-loss_0.566-acc_0.792-val_loss_0.4294-val_acc_0.8176-val_fmeasure_0.7883-val_mcc_0.6525-val_mean_acc_0.8376.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV5/model.0182-loss_0.620-acc_0.755-val_loss_0.5914-val_acc_0.6783-val_fmeasure_0.5400-val_mcc_0.3362-val_mean_acc_0.6886.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV6/model.0149-loss_0.625-acc_0.760-val_loss_0.4992-val_acc_0.7415-val_fmeasure_0.7031-val_mcc_0.4924-val_mean_acc_0.7534.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV7/model.0164-loss_0.610-acc_0.763-val_loss_0.5381-val_acc_0.7315-val_fmeasure_0.7015-val_mcc_0.4643-val_mean_acc_0.7354.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV8/model.0740-loss_0.549-acc_0.793-val_loss_0.4391-val_acc_0.8203-val_fmeasure_0.7965-val_mcc_0.6357-val_mean_acc_0.8185.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV9/model.0253-loss_0.633-acc_0.733-val_loss_0.4939-val_acc_0.7721-val_fmeasure_0.7669-val_mcc_0.5440-val_mean_acc_0.7719.h5', '/home/mhubrich/checkpoints/adni/meanROI1_5_pretrained_CV10/model.0711-loss_0.558-acc_0.799-val_loss_0.4784-val_acc_0.7625-val_fmeasure_0.7500-val_mcc_0.5356-val_mean_acc_0.7702.h5']

def train():
    # Get inputs for training and validation
    scans_train = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_train')
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_val')
    train_inputs = inputs(scans_train, target_size, batch_size, load_all_scans, classes, 'train', SEED, 'binary')
    val_inputs = inputs(scans_val, target_size, batch_size, load_all_scans, classes, 'predict', SEED, 'binary')

    # Set up the model
    if path_weights is None:
        model = build_model(1)
        sgd = SGD(lr=0.0005, decay=0.0005, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
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
            Evaluation(val_inputs,
                       [callbacks.early_stop(patience=50, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']),
                        callbacks.save_model(path_checkpoints, max_files=3, monitor=['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc'])])]

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

