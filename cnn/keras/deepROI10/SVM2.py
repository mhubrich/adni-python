##############################################################
# Set seed for determinisitc behaviour between different runs.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import pickle

from utils.load_scans import load_scans
from utils.sort_scans import sort_groups
from utils.split_scans import read_imageID
import sys


# Fold of pretrained CNN
pre = str(sys.argv[1])

# Grid search for RBF-SVM
C = [2**0, 2**2, 2**4, 2**6, 2**8]
gamma = [2**(-12), 2**(-10), 2**(-8), 2**(-6)]

# Threshold for importance map
t = 0.00001

# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_predictions = 'predictions/predictions_deepROI12_fliter_'


def get_indices(set):
    indices = []
    for s in set:
        for i in range(len(scans)):
            if s.imageID == scans[i].imageID:
                indices.append(i)
                break
    return indices


def get_importanceMap(importanceMap_all, scans, t=0.000001):
    importanceMap = np.zeros((22, 22, 22), dtype=np.float32)
    for s in scans:
        importanceMap += importanceMap_all[s.imageID]
    importanceMap /= len(scans)
    importanceMap[np.where(importanceMap < t)] = 0
    importanceMap[np.where(importanceMap >= t)] = 1
    return importanceMap


def get_scans(path):
    return read_imageID(scans, path)


def get_data(scans, importanceMap):
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

importanceMap_all = pickle.load(open('importanceMap_all_deepROI12_fliter_135', 'rb'))
for fold in range(1, 11):
    scans_train = get_scans('/home/mhubrich/rui_li_scans/' + str(fold) + '_train')
    scans_val = get_scans('/home/mhubrich/rui_li_scans/' + str(fold) + '_val')
    scans_test = get_scans('/home/mhubrich/rui_li_scans/' + str(fold) + '_test')
    importanceMap = get_importanceMap(importanceMap_all, scans_train + scans_val, t)
    print(np.count_nonzero(importanceMap))
    X_train, y_train = get_data(scans_train, importanceMap)
    X_val, y_val = get_data(scans_val, importanceMap)
    X_test, y_test = get_data(scans_test, importanceMap)

    for g in range(len(gamma)):
        for c in range(len(C)):
            clf = svm.SVC(C=C[c], gamma=gamma[g], kernel='rbf', random_state=SEED)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            evaluate(metrics_val, y_val, y_pred, fold-1, g, c)

            y_pred = clf.predict(X_test)
            evaluate(metrics_test, y_test, y_pred, fold-1, g, c)


np.save('SVM/metrics_val_deepROI10_' + pre + '_ADNC_00000003_135_stnd.npy', metrics_val)
np.save('SVM/metrics_test_deepROI10_' + pre + '_ADNC_00000003_135_stnd.npy', metrics_test)


