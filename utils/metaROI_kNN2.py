import os
import random
import numpy as np
import nibabel as nib
from sklearn.neighbors import NearestNeighbors

from utils.sort_scans import sort_groups
from utils.load_scans import Scan


scan_shape = (91, 109, 91)
path_feature_means = '/home/mhubrich/metaROI/metaROI_features_means.npy'
path_feature_names = '/home/mhubrich/metaROI/metaROI_features_names.npy'
base_roi = '/home/mhubrich/metaROI'

rois = ['LAngular.nii', 'LTemporal.nii', 'PostCing.nii', 'RAngular.nii', 'RTemporal.nii']
directories = ['metaROI1', 'metaROI2', 'metaROI3', 'metaROI4', 'metaROI5']
ranges = [(range(65, 73), range(28, 36), range(51, 59)), (range(73, 79), range(39, 45), range(27, 33)),
          (range(39, 49), range(37, 47), range(46, 56)), (range(13, 26), range(28, 41), range(47, 60)),
          (range(11, 16), range(35, 40), range(31, 36))]
ROI = []
for i in range(len(rois)):
    ROI.append(nib.load(os.path.join(base_roi, rois[i])).get_data()[ranges[i][0],:,:][:,ranges[i][1],:][:,:,ranges[i][2]])


def save_metaROIs(mean_scan, i, name, group, save_dir):
    if not os.path.exists(os.path.join(save_dir, directories[i])):
        os.makedirs(os.path.join(save_dir, directories[i]))
    s = mean_scan[ranges[i][0],:,:][:,ranges[i][1],:][:,:,ranges[i][2]] * ROI[i]
    np.save(os.path.join(save_dir, directories[i], 'mean_' + name + '.npy'), s)
    return Scan('Artificial_'+name, name, None, None, group, None, None,
                os.path.join(save_dir, directories[i], 'mean_' + name + '.npy'))


def mean_scans(paths, indices):
    mean_scan = np.zeros(scan_shape, dtype=np.float32)
    for i in indices:
        scan = nib.load(paths[i]).get_data()
        scan -= np.min(scan)
        scan /= (np.max(scan) - np.min(scan))
        mean_scan += scan
    mean_scan /= len(indices)
    return mean_scan


def kNN(scans, k, amount, group, save_dir, max_dist=0.01, seed=None):
    np.random.seed(seed)
    means = np.load(path_feature_means)
    imageIDs = np.load(path_feature_names)
    groups, _ = sort_groups(scans)
    indices = []
    paths = []
    for i in range(len(imageIDs)):
        for scan in groups[group]:
            if imageIDs[i] == scan.imageID:
                indices.append(i)
                # TODO WORKAROUND
                #paths.append(scan.path)
                p = scan.path
                p = p.replace('.npy', '.nii')
                p = p.replace('ADNI_intnorm_metaROI1', 'ADNI_intnorm')
                paths.append(p)
    assert amount < len(indices), \
        "There are less scans to sample from as the specified amount."
    created_scans = []
    samples = random.sample(indices, amount)
    nbrs = []
    for i in range(len(ROI)):
        nbrs.append(NearestNeighbors(n_neighbors=k).fit(means[indices, i].reshape(-1, 1)))
    for sample in samples:
        tmp = []
        name = imageIDs[sample]
        flag = True
        for i in range(len(ROI)):
            distances, ind = nbrs[i].kneighbors(means[sample, i])
            if np.max(distances) > max_dist:
                flag = False
                break
            mean_scan = mean_scans(paths, ind[0])
            tmp.append(save_metaROIs(mean_scan, i, name, group, save_dir))
        if flag:
            created_scans.append(tmp[0])
    return created_scans

