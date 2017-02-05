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


# Grid search for RBF-SVM
C = [2**0, 2**2, 2**4, 2**6, 2**8]
gamma = [2**(-12), 2**(-10), 2**(-8), 2**(-6)]

# Paths
path_ADNI = '/home/mhubrich/ADNI_pSMC_deepROI6_1'

scans = read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/rui_li_NC_imageIDs')
scans += read_imageID(path_ADNI, '/home/mhubrich/rui_li_scans/rui_li_AD_imageIDs')

data = np.zeros((len(scans), 22*22*22), dtype=np.float32)
labels = np.zeros(len(scans), dtype=np.int32)

for i in range(len(scans)):
    data[i] = np.load(scans[i].path).flatten()
    labels[i] = 0 if scans[i].group == 'Normal' else 1

metrics_val = np.zeros((len(scans), len(gamma), len(C)), dtype=np.float32)
metrics_test = np.zeros((len(scans), len(gamma), len(C)), dtype=np.float32)

indices = list(np.arange(len(scans)))
for i in range(len(scans)):
    LOO = indices[:i]+indices[i+1:]
    train, val = train_test_split(LOO, stratify=labels[LOO], test_size=0.2, random_state=SEED+i)
    X_train = data[train]
    y_train = labels[train]
    X_val = data[val]
    y_val = labels[val]
    X_test = np.expand_dims(data[i], 0)
    y_test = [labels[i]]
    for g in range(len(gamma)):
        for c in range(len(C)):
            clf = svm.SVC(C=C[c], gamma=gamma[g], kernel='rbf', random_state=SEED)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            metrics_val[i, g, c] = accuracy_score(y_val, np.round(y_pred))

            y_pred = clf.predict(X_test)
            metrics_test[i, g, c] = accuracy_score(y_test, np.round(y_pred))


np.save('SVM/metrics_val_deepROI10_baseline_LOO.npy', metrics_val)
np.save('SVM/metrics_test_deepROI10_baseline_LOO.npy', metrics_test)
np.save('SVM/metrics_labels_deepROI10_baseline_LOO.npy', labels)

