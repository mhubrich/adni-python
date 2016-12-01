import os
import numpy
#from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.load_scans import load_scans
from utils.sort_scans import sort_subjects


def read_imageID(path_ADNI, fname):
    ret = []
    with open(fname, 'rb') as f:
        content = [x.strip(os.linesep) for x in f.readlines()]
    scans = load_scans(path_ADNI)
    for imageID in content:
        for scan in scans:
            if imageID == scan.imageID:
                ret.append(scan)
                break
    return ret


def write_imageID(scans, fname):
    with open(fname, 'wb') as f:
        for scan in scans:
            f.write(scan.imageID + os.linesep)


def CV(scans, k, val_split, classes, path, seed=None):
    class_indices = dict(zip(classes, np.arange(len(classes))))
    subjects, subject_names = sort_subjects(scans)
    x, y = [], []
    for n in subject_names:
        counts = {}
        flag = False
        age = 0.0
        for scan in subjects[n]:
            if scan.group in classes:
                flag = True
                age += scan.age
                if scan.group in counts:
                    counts[scan.group] += 1
                else:
                    counts[scan.group] = 1
        age /= np.sum(counts.values())
        if flag:
            x.append(n)
            if age <= 76:
                y.append(class_indices[max(counts, key=counts.get)])
            else:
                y.append(class_indices[max(counts, key=counts.get)] + len(classes))
    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    fold = 0
    for index, test_index in skf.split(x, y):
        fold += 1
        # First, use set 'index' to generate train and val sets
        y_index = [y[i] for i in index]
        train_index, val_index = train_test_split(index, stratify=y_index, test_size=val_split, random_state=seed)
        # Save train, val and test sets
        write_imageID([s for i in train_index for s in subjects[x[i]]], os.path.join(path, str(fold) + '_train'))
        write_imageID([s for i in val_index for s in subjects[x[i]]], os.path.join(path, str(fold) + '_val'))
        write_imageID([s for i in test_index for s in subjects[x[i]]], os.path.join(path, str(fold) + '_test'))

