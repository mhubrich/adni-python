import fnmatch
import os
import numpy as np


directory = '/home/mhubrich/ADNI_intnorm_deepROI5_3'
importanceMap = np.load('importanceMap_2_45.npy')

importanceMap[np.where(importanceMap <= 0)] = 0
importanceMap[np.where(importanceMap > 0)] = 1


files = []
for root, dirnames, filenames in os.walk(directory):
    for filename in fnmatch.filter(filenames, '*.npy'):
        files.append(os.path.join(root, filename))


for f in files:
    scan = np.load(f)
    scan *= importanceMap
    scan[np.where(scan < 0)] = 0
    np.save(f, scan)

