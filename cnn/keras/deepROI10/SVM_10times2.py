##############################################################
# Set seed for determinisitc behaviour between different runs.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import RFE

from utils.load_scans import load_scans
from utils.sort_scans import sort_groups
from utils.split_scans import read_imageID
from cnn.keras.deepROI10.create_importanceMap_full_all2 import create_importanceMap
import sys


# Fold of pretrained CNN
pre = str(sys.argv[1])

# Grid search for RBF-SVM
C = [2**0, 2**2, 2**4, 2**6, 2**8]
gamma = [2**(-10), 2**(-8), 2**(-6)]

# Threshold importance map
t = 0

# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_predictions = 'predictions/predictions_deepROI10_fliter_'


scans = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/rui_li_NC_imageIDs')
scans += read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/rui_li_AD_imageIDs')

data = np.zeros((len(scans), 22, 22, 22), dtype=np.float32)
labels = np.zeros(len(scans), dtype=np.int32)

for i in range(len(scans)):
    data[i] = np.load(scans[i].path)
    labels[i] = 0 if scans[i].group == 'Normal' else 1


def get_indices(set):
    indices = []
    for s in set:
        for i in range(len(scans)):
            if s.imageID == scans[i].imageID:
                indices.append(i)
                break
    return indices


def get_data(scans, importanceMap):
    indices = get_indices(scans)
    X = np.zeros((len(indices), np.count_nonzero(importanceMap)), dtype=np.float32)
    for i in range(len(indices)):
        X[i] = data[indices[i]][np.where(importanceMap > 0)]
    y = labels[indices]
    return X, y


def evaluate(a, y, y_pred, i, fold, g, c):
    if len(np.unique(y) == 1):
        if np.sum(y) == 0:
            tp = 1.0
            tn = 1.0 - (np.sum(y_pred) / float(len(y_pred)))
        else:
            tn = 1.0
            tp = np.sum(y_pred) / float(len(y_pred))
    else:
        conf = confusion_matrix(y, np.round(y_pred))
        tn = float(conf[0,0]) / (conf[0,0] + conf[0,1])
        tp = float(conf[1,1]) / (conf[1,1] + conf[1,0])
    a[i, fold, 0, g, c] = accuracy_score(y, np.round(y_pred))
    a[i, fold, 1, g, c] = (tn + tp) / 2.0
    a[i, fold, 2, g, c] = tn
    a[i, fold, 3, g, c] = tp


metrics_val = np.zeros((10, 10, 4, len(gamma), len(C)), dtype=np.float32)
metrics_test = np.zeros((10, 10, 4, len(gamma), len(C)), dtype=np.float32)

importanceMap_all = create_importanceMap([1, 3, 5], pre, path_predictions)

for i in range(1, 11):
    for fold in range(1, 11):
        scans_train = read_imageID(scans, '/home/mhubrich/rui_10times/' + str(i) + '_' + str(fold) + '_train')
        scans_val = read_imageID(scans, '/home/mhubrich/rui_10times/' + str(i) + '_' + str(fold) + '_val')
        scans_test = read_imageID(scans, '/home/mhubrich/rui_10times/' + str(i) + '_' + str(fold) + '_test')
        importanceMap = [np.zeros((22, 22, 22), dtype=np.float32), np.zeros((22, 22, 22), dtype=np.float32)]
        for s in scans_train + scans_val:
            importanceMap[0] += importanceMap_all[s.imageID][0]
            importanceMap[1] += importanceMap_all[s.imageID][1]
        importanceMap[0] /= len(scans_train + scans_val)
        importanceMap[1] /= len(scans_train + scans_val)
        importanceMap[0][np.where(importanceMap[0] < t)] = 0
        importanceMap[0][np.where(importanceMap[0] > t)] = 1
        importanceMap[1][np.where(importanceMap[1] < t)] = 0
        importanceMap[1][np.where(importanceMap[1] > t)] = 1
        importanceMap = importanceMap[0] + importanceMap[1]
        X_train, y_train = get_data(scans_train, importanceMap)
        X_val, y_val = get_data(scans_val, importanceMap)
        X_test, y_test = get_data(scans_test, importanceMap)
        for g in range(len(gamma)):
            for c in range(len(C)):
                clf = svm.SVC(C=C[c], kernel='linear', random_state=SEED)
                selector = RFE(clf, None, step=0.01)
                selector.fit(X_train, y_train)
                print('done fitting')

                y_pred = clf.predict(X_val)
                evaluate(metrics_val, y_val, y_pred, i-1, fold-1, g, c)

                y_pred = clf.predict(X_test)
                evaluate(metrics_test, y_test, y_pred, i-1, fold-1, g, c)
    print(i)


np.save('SVM/metrics_val_deepROI10_' + pre + '_0_135_full2_10times.npy', metrics_val)
np.save('SVM/metrics_test_deepROI10_' + pre + '_0_135_full2_10times.npy', metrics_test)

