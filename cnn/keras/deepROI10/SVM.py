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
import sys


pre = str(sys.argv[1])

def precision(y_true, y_pred):
    return TP(y_true, y_pred) / (TP(y_true, y_pred) + 1.0 - TN(y_true, y_pred))


C = [1, 10, 100]
gamma = [0.01, 0.025, 0.05, 0.075]

path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'
path_importanceMap = 'importanceMap_2_35_fold_'
scans = []
for i in range(1, 11):
    scans += read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(i) + '_test')

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


metrics_val = np.zeros((10, 7, len(gamma), len(C)), dtype=np.float32)
metrics_test = np.zeros((10, 7, len(gamma), len(C)), dtype=np.float32)

for g in range(len(gamma)):
    for c in range(len(C)):
        for fold in range(1, 11):
            importanceMap_NC = np.load(path_importanceMap + str(fold) + '_' +  pre + '_NC.npy')
            importanceMap_NC[np.where(importanceMap_NC < 0.01)] = 0
            importanceMap_NC[np.where(importanceMap_NC >= 0.01)] = 1
            importanceMap_AD = np.load(path_importanceMap + str(fold) + '_' + pre + '_AD.npy')
            importanceMap_AD[np.where(importanceMap_AD < 0.01)] = 0
            importanceMap_AD[np.where(importanceMap_AD >= 0.01)] = 1
            scans_train = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(fold) + '_train')
            indices = get_indices(scans_train)
            X = np.zeros((len(indices), np.count_nonzero(importanceMap_NC) + np.count_nonzero(importanceMap_AD)), dtype=np.float32)
            for i in range(len(indices)):
                X[i] = np.concatenate((data[indices[i]][np.where(importanceMap_NC > 0)], data[indices[i]][np.where(importanceMap_AD > 0)]), axis=0)
            y = labels[indices]

            clf = svm.SVC(C=C[c], gamma=gamma[g], kernel='rbf', random_state=SEED)
            clf.fit(X, y)

            scans_val = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/' + str(fold) + '_val')
            indices = get_indices(scans_val)
            X = np.zeros((len(indices), np.count_nonzero(importanceMap_NC) + np.count_nonzero(importanceMap_AD)), dtype=np.float32)
            for i in range(len(indices)):
                X[i] = np.concatenate((data[indices[i]][np.where(importanceMap_NC > 0)], data[indices[i]][np.where(importanceMap_AD > 0)]), axis=0)
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
            X = np.zeros((len(indices), np.count_nonzero(importanceMap_NC) + np.count_nonzero(importanceMap_AD)), dtype=np.float32)
            for i in range(len(indices)):
                X[i] = np.concatenate((data[indices[i]][np.where(importanceMap_NC > 0)], data[indices[i]][np.where(importanceMap_AD > 0)]), axis=0)
            y = labels[indices]

            y_pred = clf.predict(X)
            metrics_test[fold-1, 0, g, c] = accuracy(y, y_pred)
            metrics_test[fold-1, 1, g, c] = mean_accuracy(y, y_pred)
            metrics_test[fold-1, 2, g, c] = TN(y, y_pred)
            metrics_test[fold-1, 3, g, c] = TP(y, y_pred)
            metrics_test[fold-1, 4, g, c] = precision(y, y_pred)
            metrics_test[fold-1, 5, g, c] = fmeasure(y, y_pred)
            metrics_test[fold-1, 6, g, c] = roc_auc_score(y, y_pred)


np.save('metrics_val_deepROI10_' + pre + '_01_2.npy', metrics_val)
np.save('metrics_test_deepROI10_' + pre + '_01_2.npy', metrics_test)

