import os
import numpy as np
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


def CV_AAL2(scans, k, val_split, classes, path, seed=None):
    num_intervals = 2
    indices = ['AAL64', 'AAL65', 'AAL34', 'AAL35', 'AAL61']
    class_indices = dict(zip(classes, np.arange(len(classes))))
    means = {}
    subjects, subject_names = sort_subjects(scans)
    for n in subject_names:
        counts = {}
        m = 0.0
        flag = False
        for scan in subjects[n]:
            if scan.group in classes:
                flag = True
                if scan.group in counts:
                    counts[scan.group] += 1
                else:
                    counts[scan.group] = 1
                for j in range(len(indices)):
                    m += np.mean(np.load(scan.path.replace('AAL64', indices[j])))
        if flag:
            m /= np.sum(counts.values()) * len(indices)
            if max(counts, key=counts.get) in means:
                means[max(counts, key=counts.get)].append(m)
            else:
                means[max(counts, key=counts.get)] = [m]
    ranges = {}
    for key in means:
        ranges[key] = []
        for j in range(1, num_intervals):
        #    ranges[key].append(np.sort(means[key])[(len(means[key])/num_intervals) * j])
            ranges[key].append(np.min(means[key]) + j * ((np.max(means[key]) - np.min(means[key]))/num_intervals))
    x, y = [], []
    for n in subject_names:
        counts = {}
        m = 0.0
        age = 0.0
        flag = False
        for scan in subjects[n]:
            if scan.group in classes:
                flag = True
                age += scan.age
                if scan.group in counts:
                    counts[scan.group] += 1
                else:
                    counts[scan.group] = 1
                for j in range(len(indices)):
                    m += np.mean(np.load(scan.path.replace('AAL64', indices[j])))
        if flag:
            x.append(n)
            age /= np.sum(counts.values())
            m /= np.sum(counts.values()) * len(indices)
            interval = len(ranges[max(counts, key=counts.get)])
            for i in range(len(ranges[max(counts, key=counts.get)])):
                if m < ranges[max(counts, key=counts.get)][i]:
                    interval = i
                    break
            label = str(interval)
            if age <= 75.5:
                label += '0'
            else:
                label += '1'
            if subjects[n][0].gender == 'M':
                label += '0'
            else:
                label += '1'
            y.append(class_indices[max(counts, key=counts.get)] * int(3*str(num_intervals)) + int(label))
    return x, y
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



def CV_AAL(scans, k, val_split, classes, path, seed=None):
    num_intervals = 5
    indices = ['AAL64', 'AAL65', 'AAL34', 'AAL35', 'AAL61']
    class_indices = dict(zip(classes, np.arange(len(classes))))
    means = {}
    subjects, subject_names = sort_subjects(scans)
    for n in subject_names:
        counts = {}
        m = 0.0
        flag = False
        for scan in subjects[n]:
            if scan.group in classes:
                flag = True
                if scan.group in counts:
                    counts[scan.group] += 1
                else:
                    counts[scan.group] = 1
                for j in range(len(indices)):
                    m += np.mean(np.load(scan.path.replace('AAL64', indices[j])))
        if flag:
            m /= np.sum(counts.values()) * len(indices)
            if max(counts, key=counts.get) in means:
                means[max(counts, key=counts.get)].append(m)
            else:
                means[max(counts, key=counts.get)] = [m]
    ranges = {}
    for key in means:
        ranges[key] = []
        for j in range(1, num_intervals):
            ranges[key].append(np.sort(means[key])[(len(means[key])/num_intervals) * j])
    x, y = [], []
    for n in subject_names:
        counts = {}
        m = 0.0
        flag = False
        for scan in subjects[n]:
            if scan.group in classes:
                flag = True
                if scan.group in counts:
                    counts[scan.group] += 1
                else:
                    counts[scan.group] = 1
                for j in range(len(indices)):
                    m += np.mean(np.load(scan.path.replace('AAL64', indices[j])))
        if flag:
            x.append(n)
            m /= np.sum(counts.values()) * len(indices)
            interval = len(ranges[max(counts, key=counts.get)])
            for i in range(len(ranges[max(counts, key=counts.get)])):
                if m < ranges[max(counts, key=counts.get)][i]:
                    interval = i
                    break
            y.append(class_indices[max(counts, key=counts.get)] * num_intervals + interval)
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
        if flag:
            x.append(n)
            age /= np.sum(counts.values())
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


def split(scans, val_split, classes, path, seed=None):
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
        if flag:
            x.append(n)
            age /= np.sum(counts.values())
            if age <= 76:
                y.append(class_indices[max(counts, key=counts.get)])
            else:
                y.append(class_indices[max(counts, key=counts.get)] + len(classes))
    train_index, val_index = train_test_split(np.arange(len(x)), stratify=y, test_size=val_split, random_state=seed)
    # Save train and val set
    write_imageID([s for i in train_index for s in subjects[x[i]]], os.path.join(path, 'train'))
    write_imageID([s for i in val_index for s in subjects[x[i]]], os.path.join(path, 'val'))

