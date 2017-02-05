##############################################################
# Set seed for determinisitc behaviour between different runs.
SEED = 0
import numpy as np
np.random.seed(SEED)
##############################################################

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

from utils.load_scans import load_scans
from utils.sort_scans import sort_groups
from utils.split_scans import read_imageID
from cnn.keras.deepROI10.create_importanceMap_full_all import create_importanceMap
import sys


# Fold of pretrained CNN
pre = str(sys.argv[1])

# Grid search for RBF-SVM
C = [2**0, 2**2, 2**4, 2**6]
gamma = [2**(-12), 2**(-10), 2**(-8)]

# Threshold importance map
t = 0.000000

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

metrics_val = np.zeros((len(scans), len(gamma), len(C)), dtype=np.float32)
metrics_test = np.zeros((len(scans), len(gamma), len(C)), dtype=np.float32)

importanceMap_all = create_importanceMap([1, 3, 5], pre, path_predictions)
indices = list(np.arange(len(scans)))
for i in range(len(scans)):
    LOO = indices[:i]+indices[i+1:]
    importanceMap = np.zeros((22, 22, 22), dtype=np.float32)
    for j in LOO:
        importanceMap += importanceMap_all[scans[j].imageID]
    importanceMap /= len(LOO)
    importanceMap[np.where(importanceMap < t)] = 0
    importanceMap[np.where(importanceMap >= t)] = 1
    train, val = train_test_split(LOO, stratify=labels[LOO], test_size=0.2, random_state=SEED+i)
    X_train = np.zeros((len(train), np.count_nonzero(importanceMap)), dtype=np.float32)
    for j in range(len(train)):
        X_train[j] = data[j][np.where(importanceMap > 0)]
    y_train = labels[train]
    X_val = np.zeros((len(val), np.count_nonzero(importanceMap)), dtype=np.float32)
    for j in range(len(val)):
        X_val[j] = data[j][np.where(importanceMap > 0)]
    y_val = labels[val]
    X_test = np.expand_dims(data[i][np.where(importanceMap > 0)], 0)
    y_test = [labels[i]]

    for g in range(len(gamma)):
        for c in range(len(C)):
            clf = svm.SVC(C=C[c], gamma=gamma[g], kernel='rbf', random_state=SEED)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            metrics_val[i, g, c] = accuracy_score(y_val, np.round(y_pred))

            y_pred = clf.predict(X_test)
            metrics_test[i, g, c] = accuracy_score(y_test, np.round(y_pred))


np.save('SVM/metrics_val_deepROI10_' + pre + '_000005_135_full_LOO.npy', metrics_val)
np.save('SVM/metrics_test_deepROI10_' + pre + '_000005_135_full_LOO.npy', metrics_test)
np.save('SVM/metrics_labels_deepROI10_' + pre + '_000005_135_full_LOO.npy', labels)

