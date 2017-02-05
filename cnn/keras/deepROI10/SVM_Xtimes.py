##############################################################
# Set seed for determinisitc behaviour between different runs.
import sys
import numpy as np
### Number of Fold
time = str(sys.argv[1])
###
SEED = int(time)
np.random.seed(SEED)
##############################################################

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix

from utils.load_scans import load_scans
from utils.sort_scans import sort_groups
from utils.split_scans import read_imageID


# Fold of pretrained CNN
pre = str(sys.argv[2])

# Grid search for RBF-SVM
C = [2**0, 2**2, 2**4, 2**6, 2**8]
gamma = [2**(-12), 2**(-10), 2**(-8), 2**(-6)]

# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = 'importanceMaps/importanceMap_2_135_fold_'


def get_indices(set):
    indices = []
    for s in set:
        for i in range(len(scans)):
            if s.imageID == scans[i].imageID:
                indices.append(i)
                break
    return indices


def get_importanceMap(path, t=0.00001):
    importanceMap = np.load(path)
    importanceMap[np.where(importanceMap < t)] = 0
    importanceMap[np.where(importanceMap >= t)] = 1
    return importanceMap


def get_data(path, importanceMap):
    scans = read_imageID(path_ADNI, path)
    indices = get_indices(scans)
    X = np.zeros((len(indices), np.count_nonzero(importanceMap)), dtype=np.float32)
    for i in range(len(indices)):
        X[i] = data[indices[i]][np.where(importanceMap > 0)]
    y = labels[indices]
    return X, y


def evaluate(a, y, y_pred, fold, g, c):
    conf = confusion_matrix(y, np.round(y_pred))
    a[fold, 0, g, c] = accuracy_score(y, np.round(y_pred))
    a[fold, 2, g, c] = float(conf[0,0]) / (conf[0,0] + conf[0,1])
    a[fold, 3, g, c] = float(conf[1,1]) / (conf[1,1] + conf[1,0])
    a[fold, 1, g, c] = (a[fold, 2, g, c] + a[fold, 3, g, c]) / 2.0
    a[fold, 4, g, c] = precision_score(y, np.round(y_pred))
    a[fold, 5, g, c] = f1_score(y, np.round(y_pred))
    a[fold, 6, g, c] = roc_auc_score(y, y_pred)


scans = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/rui_li_NC_imageIDs')
scans += read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/rui_li_AD_imageIDs')

data = np.zeros((len(scans), 22, 22, 22), dtype=np.float32)
labels = np.zeros(len(scans), dtype=np.int32)

for i in range(len(scans)):
    data[i] = np.load(scans[i].path)
    labels[i] = 0 if scans[i].group == 'Normal' else 1

metrics_val = np.zeros((10, 7, len(gamma), len(C)), dtype=np.float32)
metrics_test = np.zeros((10, 7, len(gamma), len(C)), dtype=np.float32)

for g in range(len(gamma)):
    for c in range(len(C)):
        for fold in range(1, 11):
            importanceMap = get_importanceMap(path_importanceMap + str(fold) + '_' + pre + '_full.npy', t=0.000005)

            X, y = get_data('/home/mhubrich/rui_10times/' + time + '_' + str(fold) + '_train', importanceMap)
            clf = svm.SVC(C=C[c], gamma=gamma[g], kernel='rbf', random_state=SEED)
            clf.fit(X, y)

            X, y = get_data('/home/mhubrich/rui_10times/' + time + '_' + str(fold) + '_val', importanceMap)
            y_pred = clf.predict(X)
            evaluate(metrics_val, y, y_pred, fold-1, g, c)

            X, y = get_data('/home/mhubrich/rui_10times/' + time + '_' + str(fold) + '_test', importanceMap)
            y_pred = clf.predict(X)
            evaluate(metrics_test, y, y_pred, fold-1, g, c)


np.save('SVM/metrics_val_deepROI10_' + pre + '_000005_135_full_' + time + '.npy', metrics_val)
np.save('SVM/metrics_test_deepROI10_' + pre + '_000005_135_full_' + time + '.npy', metrics_test)


