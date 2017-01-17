import numpy as np
import nibabel as nib
import csv


target_size = (22, 22, 22)
filter_length = [1, 3, 5]
classes = {'Normal':0, 'AD':1}

path_predictions = 'predictions_deepROI6_1_fliter_'

importanceMap = np.zeros(target_size, dtype=np.float32)
importanceCounts = np.zeros(target_size, dtype=np.int32)

for k in filter_length:
    predictions = []
    imageIDs = {}
    with open(path_predictions + str(k) + '.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if int(row[2]) == -1 and int(row[3]) == -1 and int(row[4]) == -1:
                imageIDs[row[0]] = float(row[5])
            else:
                predictions.append([row[0], classes[row[1]], int(row[2]), int(row[3]), int(row[4]), float(row[5])])

    for i in range(len(predictions)):
        if np.round(imageIDs[predictions[i][0]]) != predictions[i][1]:
            continue
        importance = predictions[i][5] - imageIDs[predictions[i][0]]
        if predictions[i][1] == classes['AD']:
            importance *= -1
        x = predictions[i][2]
        y = predictions[i][3]
        z = predictions[i][4]
        importanceMap[x, y, z] += importance
        importanceCounts[x, y, z] += 1


importanceMap /= importanceCounts

np.save('importanceMap_1_135_correct.npy', importanceMap)

img = nib.Nifti1Image(importanceMap, np.eye(4))
nib.save(img, 'importanceMap_1_135_correct.nii')


