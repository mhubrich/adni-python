##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from keras.optimizers import SGD
from cnn.keras.models.AVG444.model import build_model
from cnn.keras.AVG444.image_processing import inputs
from utils.split_scans import read_imageID
from utils.sort_scans import sort_groups
import sys

fold = str(sys.argv[1])
#sys.stdout = sys.stderr = open('output_1_' + fold, 'w')

# Training specific parameters
target_size = (22, 22, 22)
classes = ['Normal', 'AD']
batch_size = 64
load_all_scans = True
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_avgpool444'
path_checkpoints = ['/home/mhubrich/checkpoints/adni/AVG444_1_CV1/model.0357-loss_0.309-acc_0.966-val_loss_0.4842-val_acc_0.8794-val_fmeasure_0.8794-val_mcc_0.7590-val_mean_acc_0.8795.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV2/model.0159-loss_0.550-acc_0.878-val_loss_0.3468-val_acc_0.8889-val_fmeasure_0.8931-val_mcc_0.7781-val_mean_acc_0.8885.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV3/model.0440-loss_0.488-acc_0.901-val_loss_0.2656-val_acc_0.9275-val_fmeasure_0.9265-val_mcc_0.8566-val_mean_acc_0.9279.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV4/model.0297-loss_0.395-acc_0.936-val_loss_0.2155-val_acc_0.9434-val_fmeasure_0.9441-val_mcc_0.8886-val_mean_acc_0.9447.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV5/model.0241-loss_0.433-acc_0.922-val_loss_0.3891-val_acc_0.8811-val_fmeasure_0.8661-val_mcc_0.7600-val_mean_acc_0.8782.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV6/model.0184-loss_0.486-acc_0.904-val_loss_0.2749-val_acc_0.9184-val_fmeasure_0.9189-val_mcc_0.8381-val_mean_acc_0.9192.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV7/model.0232-loss_0.454-acc_0.913-val_loss_0.1351-val_acc_0.9664-val_fmeasure_0.9660-val_mcc_0.9337-val_mean_acc_0.9666.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV8/model.0187-loss_0.508-acc_0.902-val_loss_0.2162-val_acc_0.9375-val_fmeasure_0.9273-val_mcc_0.8744-val_mean_acc_0.9411.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV9/model.0274-loss_0.431-acc_0.924-val_loss_0.2381-val_acc_0.9485-val_fmeasure_0.9489-val_mcc_0.8996-val_mean_acc_0.9501.h5', '/home/mhubrich/checkpoints/adni/AVG444_1_CV10/model.0356-loss_0.382-acc_0.937-val_loss_0.1543-val_acc_0.9688-val_fmeasure_0.9689-val_mcc_0.9394-val_mean_acc_0.9695.h5']


def predict():
    # Get inputs for training and validation
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/ADNI_CV_mean2/' + fold + '_test')

    # Set up the model
    model = build_model()
    sgd = SGD(lr=0.001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.load_weights(path_checkpoints[int(fold)-1])

    groups, _ = sort_groups(scans_val)
    import time
    for scan in groups['Normal']+groups['AD']:
        print(time.strftime('%X'))
        val_inputs = inputs(scan, target_size, batch_size, load_all_scans, classes, 'predict', SEED, 'binary')
        pred = model.predict_generator(val_inputs, val_inputs.nb_sample, max_q_size=batch_size, nb_worker=1, pickle_safe=True)


if __name__ == "__main__":
    predict()

