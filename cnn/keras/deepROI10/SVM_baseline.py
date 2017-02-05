##############################################################
# Set seed for determinisitc behaviour between different runs.
# Especially fresh weights will be initialized the same way.
# Caution: CudNN might not be deterministic after all.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from sklearn import svm

from utils.load_scans import load_scans
from utils.sort_scans import sort_groups
from utils.split_scans import read_imageID
from cnn.keras.evaluation_callback import accuracy, mean_accuracy, TP, TN, fmeasure
from sklearn.metrics import roc_auc_score



def precision(y_true, y_pred):
    return TP(y_true, y_pred) / (TP(y_true, y_pred) + 1.0 - TN(y_true, y_pred))


C = [2**0, 2**2, 2**4, 2**6, 2**8]
gamma = [2**(-14), 2**(-12), 2**(-10), 2**(-8), 2**(-6)]

path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
scans = []
for i in range(1, 11):
    scans += read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(i) + '_test')

data = np.zeros((len(scans), 22*22*22), dtype=np.float32)
labels = np.zeros(len(scans), dtype=np.int32)

for i in range(len(scans)):
    data[i] = np.load(scans[i].path).flatten()
    labels[i] = 0 if scans[i].group == 'Normal' else 1


def get_indices(set):
    indices = []
    for s in set:
        for i in range(len(scans)):
            if s.imageID == scans[i].imageID:
                indices.append(i)
                break
    return indices


metrics_val = np.zeros((10, 7, len(gamma), len(C)), dtype=np.float32)
metrics_test = np.zeros((10, 7, len(gamma), len(C)), dtype=np.float32)

for g in range(len(gamma)):
    for c in range(len(C)):
        for fold in range(1, 11):
            scans_train = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(fold) + '_train')
            indices = get_indices(scans_train)
            X = data[indices]
            y = labels[indices]

            clf = svm.SVC(C=C[c], gamma=gamma[g], kernel='rbf', random_state=SEED)
            clf.fit(X, y)

            scans_val = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(fold) + '_val')
            indices = get_indices(scans_val)
            X = data[indices]
            y = labels[indices]

            y_pred = clf.predict(X)
            metrics_val[fold-1, 0, g, c] = accuracy(y, y_pred)
            metrics_val[fold-1, 1, g, c] = mean_accuracy(y, y_pred)
            metrics_val[fold-1, 2, g, c] = TN(y, y_pred)
            metrics_val[fold-1, 3, g, c] = TP(y, y_pred)
            metrics_val[fold-1, 4, g, c] = precision(y, y_pred)
            metrics_val[fold-1, 5, g, c] = fmeasure(y, y_pred)
            metrics_val[fold-1, 6, g, c] = roc_auc_score(y, y_pred)

            scans_test = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(fold) + '_test')
            indices = get_indices(scans_test)
            X = data[indices]
            y = labels[indices]

            y_pred = clf.predict(X)
            metrics_test[fold-1, 0, g, c] = accuracy(y, y_pred)
            metrics_test[fold-1, 1, g, c] = mean_accuracy(y, y_pred)
            metrics_test[fold-1, 2, g, c] = TN(y, y_pred)
            metrics_test[fold-1, 3, g, c] = TP(y, y_pred)
            metrics_test[fold-1, 4, g, c] = precision(y, y_pred)
            metrics_test[fold-1, 5, g, c] = fmeasure(y, y_pred)
            metrics_test[fold-1, 6, g, c] = roc_auc_score(y, y_pred)

    print('Gamma: %d' % g)


np.save('metrics_val_deepROI10_baseline.npy', metrics_val)
np.save('metrics_test_deepROI10_baseline.npy', metrics_test)

