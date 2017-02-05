import sys
import numpy as np
import nibabel as nib
import csv


fold = str(sys.argv[1])
fold2 = str(sys.argv[2])

target_size = (22, 22, 22)
filter_length = [3, 5]
classes = {'Normal':0, 'AD':1}

path_predictions = 'predictions_deepROI10_2_fliter_'


def interpolate(a, importanceMap, grid, pos=(0,0,0), n=3):
    x1 = max(0, pos[0] - n/2)
    x2 = min(target_size[0], pos[0] + n/2 + 1)
    y1 = max(0, pos[1] - n/2)
    y2 = min(target_size[1], pos[1] + n/2 + 1)
    z1 = max(0, pos[2] - n/2)
    z2 = min(target_size[2], pos[2] + n/2 + 1)
    a[pos[0], pos[1], pos[2]] = np.sum(importanceMap[x1:x2, y1:y2, z1:z2]) / np.count_nonzero(grid[x1:x2, y1:y2, z1:z2])


grid = np.zeros(target_size, dtype=np.int32)
for x in range(0, target_size[0], 2):
    for y in range(0, target_size[1], 2):
        for z in range(0, target_size[2], 2):
            grid[x,y,z] = 1

indices = np.where(grid == 0)

importanceMap = np.zeros(target_size, dtype=np.float32)
counts = 0.0
for k in filter_length:
    predictions = []
    imageIDs = {}
    with open(path_predictions + str(k) + '_fold_' + fold +  '_' + fold2 + '.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if int(row[2]) == -1 and int(row[3]) == -1 and int(row[4]) == -1:
                imageIDs[row[0]] = float(row[5])
            else:
                predictions.append([row[0], classes[row[1]], int(row[2]), int(row[3]), int(row[4]), float(row[5])])

    for i in range(len(predictions)):
        if predictions[i][1] != classes['AD']:
            continue
        importance = predictions[i][5] - imageIDs[predictions[i][0]]
        if predictions[i][1] == classes['AD']:
            importance *= -1
        x = predictions[i][2]
        y = predictions[i][3]
        z = predictions[i][4]
        importanceMap[x, y, z] += importance
    counts += len(imageIDs)


importanceMap /= counts
importanceMap2 = np.array(importanceMap, copy=True)
for j in range(len(indices[0])):
    interpolate(importanceMap2, importanceMap, grid, (indices[0][j], indices[1][j], indices[2][j]))

np.save('importanceMap_2_35_fold_' + fold + '_' + fold2 + '_AD.npy', importanceMap2)


