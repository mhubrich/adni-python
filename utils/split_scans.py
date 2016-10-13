import os
import numpy as np

from utils.sort_scans import sort_groups
from utils.load_scans import load_scans


def split_scans(scans, split, names, pathOut, classes=None):
    assert sum(split) == 1.0, \
        "Sum of splits is not equal to 1.0."
    assert len(split) == len(names), \
        "Number of splits is not equal to number of provided names."

    imageID_split = []
    for _ in split:
        imageID_split.append([])

    maxValue = 999999999
    maxCounts = np.full(len(split), maxValue, dtype=np.int)

    groups, names = sort_groups(scans)
    minGroup = maxValue
    for g in groups:
        if len(g) < minGroup:
            minGroup = len(g)
    maxCounts[0] = minGroup * split[0]

    indices = np.zeros(len(groups), dtype=np.int)
    for i in range(split):
        for g in groups:
            if indices[i] < maxCounts[i]:
                maxValue += 1
    if classes is None:
        classes = names


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
