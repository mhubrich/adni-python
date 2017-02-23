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
num_epoch = 5000
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm'
path_checkpoints = '/home/mhubrich/checkpoints/adni/final_metaROI_merged_5_2_CV' + fold
#path_model = [['/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV1/model.0151-loss_0.738-acc_0.725-val_loss_0.5679-val_acc_0.7647-val_mean_acc_0.7611.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV2/model.0478-loss_0.706-acc_0.748-val_loss_0.4802-val_acc_0.8161-val_mean_acc_0.8159.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV3/model.0065-loss_0.862-acc_0.596-val_loss_0.6758-val_acc_0.8127-val_mean_acc_0.8105.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV4/model.0021-loss_0.871-acc_0.538-val_loss_0.6884-val_acc_0.8321-val_mean_acc_0.8340.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV5/model.0211-loss_0.727-acc_0.724-val_loss_0.5304-val_acc_0.8059-val_mean_acc_0.8088.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV6/model.0178-loss_0.733-acc_0.726-val_loss_0.4828-val_acc_0.8435-val_mean_acc_0.8457.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV7/model.0441-loss_0.703-acc_0.746-val_loss_0.4906-val_acc_0.8203-val_mean_acc_0.8197.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV8/model.0304-loss_0.721-acc_0.734-val_loss_0.4699-val_acc_0.8268-val_mean_acc_0.8228.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV9/model.0191-loss_0.717-acc_0.733-val_loss_0.5149-val_acc_0.8044-val_mean_acc_0.8036.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV10/model.0097-loss_0.855-acc_0.626-val_loss_0.6431-val_acc_0.8216-val_mean_acc_0.8212.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV1/model.0248-loss_0.816-acc_0.663-val_loss_0.6467-val_acc_0.7316-val_mean_acc_0.7366.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV2/model.0132-loss_0.873-acc_0.566-val_loss_0.6784-val_acc_0.7854-val_mean_acc_0.7853.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV3/model.0448-loss_0.805-acc_0.678-val_loss_0.6089-val_acc_0.7378-val_mean_acc_0.7133.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV4/model.2119-loss_0.709-acc_0.707-val_loss_0.5214-val_acc_0.7786-val_mean_acc_0.7804.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV5/model.1908-loss_0.658-acc_0.737-val_loss_0.5454-val_acc_0.7436-val_mean_acc_0.7186.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV6/model.0394-loss_0.734-acc_0.721-val_loss_0.5367-val_acc_0.7672-val_mean_acc_0.7600.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV7/model.1316-loss_0.702-acc_0.728-val_loss_0.5272-val_acc_0.7734-val_mean_acc_0.7726.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV8/model.0328-loss_0.805-acc_0.691-val_loss_0.5928-val_acc_0.7756-val_mean_acc_0.7748.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV9/model.1398-loss_0.694-acc_0.727-val_loss_0.5378-val_acc_0.7528-val_mean_acc_0.7440.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV10/model.0047-loss_0.878-acc_0.543-val_loss_0.6892-val_acc_0.7770-val_mean_acc_0.7765.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV1/model.0537-loss_0.612-acc_0.837-val_loss_0.3841-val_acc_0.8419-val_mean_acc_0.8316.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV2/model.0112-loss_0.860-acc_0.739-val_loss_0.5706-val_acc_0.8429-val_mean_acc_0.8425.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV3/model.0072-loss_0.946-acc_0.579-val_loss_0.6717-val_acc_0.8315-val_mean_acc_0.8389.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV4/model.0083-loss_0.915-acc_0.680-val_loss_0.6485-val_acc_0.7929-val_mean_acc_0.7878.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV5/model.0105-loss_0.866-acc_0.728-val_loss_0.6013-val_acc_0.8462-val_mean_acc_0.8463.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV6/model.0398-loss_0.664-acc_0.824-val_loss_0.3569-val_acc_0.8855-val_mean_acc_0.8855.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV7/model.0102-loss_0.903-acc_0.698-val_loss_0.6136-val_acc_0.8594-val_mean_acc_0.8590.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV8/model.0099-loss_0.884-acc_0.737-val_loss_0.5814-val_acc_0.8228-val_mean_acc_0.8212.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV9/model.0063-loss_0.939-acc_0.594-val_loss_0.6726-val_acc_0.8229-val_mean_acc_0.8256.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV10/model.0318-loss_0.664-acc_0.822-val_loss_0.3642-val_acc_0.8810-val_mean_acc_0.8809.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV1/model.0280-loss_0.759-acc_0.755-val_loss_0.5365-val_acc_0.7978-val_mean_acc_0.7963.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV2/model.0684-loss_0.736-acc_0.779-val_loss_0.4332-val_acc_0.8238-val_mean_acc_0.8235.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV3/model.0049-loss_0.960-acc_0.542-val_loss_0.6864-val_acc_0.7865-val_mean_acc_0.7803.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV4/model.0070-loss_0.950-acc_0.591-val_loss_0.6776-val_acc_0.8321-val_mean_acc_0.8298.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV5/model.0342-loss_0.768-acc_0.740-val_loss_0.4795-val_acc_0.8498-val_mean_acc_0.8465.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV6/model.0273-loss_0.787-acc_0.748-val_loss_0.4761-val_acc_0.8130-val_mean_acc_0.8087.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV7/model.0723-loss_0.758-acc_0.769-val_loss_0.5448-val_acc_0.7656-val_mean_acc_0.7646.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV8/model.0312-loss_0.797-acc_0.751-val_loss_0.4651-val_acc_0.8425-val_mean_acc_0.8395.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV9/model.0093-loss_0.915-acc_0.664-val_loss_0.6421-val_acc_0.7823-val_mean_acc_0.7790.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV10/model.0420-loss_0.782-acc_0.750-val_loss_0.4351-val_acc_0.8587-val_mean_acc_0.8586.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV1/model.2157-loss_0.715-acc_0.692-val_loss_0.5530-val_acc_0.7868-val_mean_acc_0.7813.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV2/model.3466-loss_0.744-acc_0.669-val_loss_0.5597-val_acc_0.7548-val_mean_acc_0.7545.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV3/model.0112-loss_0.830-acc_0.510-val_loss_0.6902-val_acc_0.7491-val_mean_acc_0.7399.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV4/model.2146-loss_0.745-acc_0.672-val_loss_0.5340-val_acc_0.7964-val_mean_acc_0.7963.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV5/model.0215-loss_0.807-acc_0.539-val_loss_0.6878-val_acc_0.7106-val_mean_acc_0.7229.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV6/model.0362-loss_0.826-acc_0.576-val_loss_0.6793-val_acc_0.7595-val_mean_acc_0.7560.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV7/model.1929-loss_0.725-acc_0.688-val_loss_0.5336-val_acc_0.7969-val_mean_acc_0.7962.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV8/model.0015-loss_0.851-acc_0.497-val_loss_0.6919-val_acc_0.8071-val_mean_acc_0.8035.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV9/model.0983-loss_0.776-acc_0.647-val_loss_0.6299-val_acc_0.7970-val_mean_acc_0.7969.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV10/model.0401-loss_0.828-acc_0.570-val_loss_0.6811-val_acc_0.7770-val_mean_acc_0.7766.h5']]
path_model = [['/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV1/model.0151-loss_0.738-acc_0.725-val_loss_0.5679-val_acc_0.7647-val_mean_acc_0.7611.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV2/model.0478-loss_0.706-acc_0.748-val_loss_0.4802-val_acc_0.8161-val_mean_acc_0.8159.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV3/model.0065-loss_0.862-acc_0.596-val_loss_0.6758-val_acc_0.8127-val_mean_acc_0.8105.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV4/model.0021-loss_0.871-acc_0.538-val_loss_0.6884-val_acc_0.8321-val_mean_acc_0.8340.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV5/model.0211-loss_0.727-acc_0.724-val_loss_0.5304-val_acc_0.8059-val_mean_acc_0.8088.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV6/model.0178-loss_0.733-acc_0.726-val_loss_0.4828-val_acc_0.8435-val_mean_acc_0.8457.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV7/model.0441-loss_0.703-acc_0.746-val_loss_0.4906-val_acc_0.8203-val_mean_acc_0.8197.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV8/model.0304-loss_0.721-acc_0.734-val_loss_0.4699-val_acc_0.8268-val_mean_acc_0.8228.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV9/model.0191-loss_0.717-acc_0.733-val_loss_0.5149-val_acc_0.8044-val_mean_acc_0.8036.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single21_1_CV10/model.0097-loss_0.855-acc_0.626-val_loss_0.6431-val_acc_0.8216-val_mean_acc_0.8212.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV1/model.0248-loss_0.816-acc_0.663-val_loss_0.6467-val_acc_0.7316-val_mean_acc_0.7366.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV2/model.0132-loss_0.873-acc_0.566-val_loss_0.6784-val_acc_0.7854-val_mean_acc_0.7853.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_2_CV3/model.1144-loss_0.763-acc_0.703-val_loss_0.5880-val_acc_0.7191-val_mean_acc_0.7211.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV4/model.2119-loss_0.709-acc_0.707-val_loss_0.5214-val_acc_0.7786-val_mean_acc_0.7804.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV5/model.1908-loss_0.658-acc_0.737-val_loss_0.5454-val_acc_0.7436-val_mean_acc_0.7186.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV6/model.0394-loss_0.734-acc_0.721-val_loss_0.5367-val_acc_0.7672-val_mean_acc_0.7600.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV7/model.1316-loss_0.702-acc_0.728-val_loss_0.5272-val_acc_0.7734-val_mean_acc_0.7726.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV8/model.0328-loss_0.805-acc_0.691-val_loss_0.5928-val_acc_0.7756-val_mean_acc_0.7748.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV9/model.1398-loss_0.694-acc_0.727-val_loss_0.5378-val_acc_0.7528-val_mean_acc_0.7440.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single22_1_CV10/model.0047-loss_0.878-acc_0.543-val_loss_0.6892-val_acc_0.7770-val_mean_acc_0.7765.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV1/model.0537-loss_0.612-acc_0.837-val_loss_0.3841-val_acc_0.8419-val_mean_acc_0.8316.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV2/model.0112-loss_0.860-acc_0.739-val_loss_0.5706-val_acc_0.8429-val_mean_acc_0.8425.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV3/model.0072-loss_0.946-acc_0.579-val_loss_0.6717-val_acc_0.8315-val_mean_acc_0.8389.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_2_CV4/model.0072-loss_0.926-acc_0.647-val_loss_0.6580-val_acc_0.8143-val_mean_acc_0.8121.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV5/model.0105-loss_0.866-acc_0.728-val_loss_0.6013-val_acc_0.8462-val_mean_acc_0.8463.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV6/model.0398-loss_0.664-acc_0.824-val_loss_0.3569-val_acc_0.8855-val_mean_acc_0.8855.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV7/model.0102-loss_0.903-acc_0.698-val_loss_0.6136-val_acc_0.8594-val_mean_acc_0.8590.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV8/model.0099-loss_0.884-acc_0.737-val_loss_0.5814-val_acc_0.8228-val_mean_acc_0.8212.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV9/model.0063-loss_0.939-acc_0.594-val_loss_0.6726-val_acc_0.8229-val_mean_acc_0.8256.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single23_1_CV10/model.0318-loss_0.664-acc_0.822-val_loss_0.3642-val_acc_0.8810-val_mean_acc_0.8809.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV1/model.0280-loss_0.759-acc_0.755-val_loss_0.5365-val_acc_0.7978-val_mean_acc_0.7963.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV2/model.0684-loss_0.736-acc_0.779-val_loss_0.4332-val_acc_0.8238-val_mean_acc_0.8235.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV3/model.0049-loss_0.960-acc_0.542-val_loss_0.6864-val_acc_0.7865-val_mean_acc_0.7803.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_2_CV4/model.0027-loss_0.957-acc_0.563-val_loss_0.6866-val_acc_0.8357-val_mean_acc_0.8332.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV5/model.0342-loss_0.768-acc_0.740-val_loss_0.4795-val_acc_0.8498-val_mean_acc_0.8465.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV6/model.0273-loss_0.787-acc_0.748-val_loss_0.4761-val_acc_0.8130-val_mean_acc_0.8087.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV7/model.0723-loss_0.758-acc_0.769-val_loss_0.5448-val_acc_0.7656-val_mean_acc_0.7646.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV8/model.0312-loss_0.797-acc_0.751-val_loss_0.4651-val_acc_0.8425-val_mean_acc_0.8395.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV9/model.0093-loss_0.915-acc_0.664-val_loss_0.6421-val_acc_0.7823-val_mean_acc_0.7790.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single24_1_CV10/model.0420-loss_0.782-acc_0.750-val_loss_0.4351-val_acc_0.8587-val_mean_acc_0.8586.h5'], ['/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV1/model.2157-loss_0.715-acc_0.692-val_loss_0.5530-val_acc_0.7868-val_mean_acc_0.7813.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV2/model.3466-loss_0.744-acc_0.669-val_loss_0.5597-val_acc_0.7548-val_mean_acc_0.7545.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_2_CV3/model.0400-loss_0.819-acc_0.568-val_loss_0.6813-val_acc_0.7378-val_mean_acc_0.7569.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV4/model.2146-loss_0.745-acc_0.672-val_loss_0.5340-val_acc_0.7964-val_mean_acc_0.7963.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV5/model.0215-loss_0.807-acc_0.539-val_loss_0.6878-val_acc_0.7106-val_mean_acc_0.7229.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV6/model.0362-loss_0.826-acc_0.576-val_loss_0.6793-val_acc_0.7595-val_mean_acc_0.7560.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV7/model.1929-loss_0.725-acc_0.688-val_loss_0.5336-val_acc_0.7969-val_mean_acc_0.7962.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV8/model.0015-loss_0.851-acc_0.497-val_loss_0.6919-val_acc_0.8071-val_mean_acc_0.8035.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV9/model.0983-loss_0.776-acc_0.647-val_loss_0.6299-val_acc_0.7970-val_mean_acc_0.7969.h5', '/home/mhubrich/checkpoints/adni/final_metaROI_single25_1_CV10/model.0401-loss_0.828-acc_0.570-val_loss_0.6811-val_acc_0.7770-val_mean_acc_0.7766.h5']]


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
    model = build_model(trainable=False)
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

