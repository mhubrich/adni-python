import numpy as np
import nibabel as nib
import csv


target_size = (22, 22, 22)
filter_length = [4, 5]
classes = {'Normal':0, 'AD':1}

path_predictions = 'predictions_deepROI5_2_fliter_'
importanceMapOld = np.load('importanceMap_1_45.npy')
if importanceMapOld is not None:
    importanceMapOld[np.where(importanceMapOld <= 0)] = 0
    importanceMapOld[np.where(importanceMapOld > 0)] = 1

importanceMap = np.zeros(target_size, dtype=np.float32)
importanceCounts = np.zeros(target_size, dtype=np.int32)

for k in filter_length:
    nb_runs = ((target_size[0]-k+1) ** 3) + 1
    predictions = []
    with open(path_predictions + str(k) + '.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            predictions.append([classes[row[1]], int(row[2]), int(row[3]), int(row[4]), float(row[5])])

    j = nb_runs - 1
    for i in range(len(predictions)):
        if (i+1) % nb_runs == 0:
            j += nb_runs
        else:
            importance = predictions[i][4] - predictions[j][4]
            if predictions[i][0] == classes['AD']:
                importance *= -1
            x = predictions[i][1]
            y = predictions[i][2]
            z = predictions[i][3]
            importanceMap[x:x+k, y:y+k, z:z+k] += importance
            importanceCounts[x:x+k, y:y+k, z:z+k] += 1


importanceMap /= importanceCounts
if importanceMapOld is not None:
    importanceMap *= importanceMapOld
np.save('importanceMap_2_45.npy', importanceMap)

img = nib.Nifti1Image(importanceMap, np.eye(4))
nib.save(img, 'importanceMap_2_45.nii')


