import fnmatch
import os
import numpy as np


def get_weights(base, k=10, monitor='val_acc'):
    paths = []
    for fold in range(1, k+1):
	if 'loss' in monitor:
	    best1 = np.Inf
	    best2 = -np.Inf
	else:
	    best1 = -np.Inf
	    best2 = np.Inf
	keep = ''
	directory = base + str(fold)
	files = []
	for root, dirnames, filenames in os.walk(directory):
	    for filename in fnmatch.filter(filenames, '*.h5'):
	        files.append(os.path.join(root, filename))
	for f in files:
	    if monitor == 'val_acc':
	        acc = float(f.split('-')[-2].split('val_acc_')[1])
	        loss = float(f.split('-')[-3].split('val_loss_')[1])
	        if np.greater(acc, best1) or (np.equal(acc, best1) and np.less(loss, best2)):
	            best1 = acc
	            best2 = loss
	            keep = f
	    elif monitor == 'val_loss':
	        acc = float(f.split('-')[-2].split('val_acc_')[1])
	        loss = float(f.split('-')[-3].split('val_loss_')[1])
	        if np.less(loss, best1) or (np.equal(loss, best1) and np.greater(acc, best2)):
	            best1 = loss
	            best2 = acc
	            keep = f
	    else:
	        mean_acc = float(f.split('-')[-1].split('val_mean_acc_')[1][:-3])
	        loss = float(f.split('-')[-3].split('val_loss_')[1])
	        if np.greater(mean_acc, best1) or (np.equal(mean_acc, best1) and np.less(loss, best2)):
	            best1 = mean_acc
	            best2 = loss
	            keep = f
	paths.append(keep)
    return paths


def get_paths(n, k=10):
    path_checkpoints = '/home/mhubrich/checkpoints/adni/deepROI9_' + str(n) + '_CV'
    return get_weights(path_checkpoints, k=k, monitor='val_loss')

