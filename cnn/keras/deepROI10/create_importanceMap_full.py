import sys
import numpy as np
import nibabel as nib
import csv


fold = str(sys.argv[1])
fold2 = str(sys.argv[2])

target_size = (22, 22, 22)
filter_length = [1, 3, 5]
classes = {'Normal':0, 'AD':1}

path_predictions = 'predictions/predictions_deepROI10_2_fliter_'


def count_voxel(pos=(0,0,0), n=3):
    x1 = max(0, pos[0] - n/2)
    x2 = min(target_size[0], pos[0] + n/2 + 1)
    y1 = max(0, pos[1] - n/2)
    y2 = min(target_size[1], pos[1] + n/2 + 1)
    z1 = max(0, pos[2] - n/2)
    z2 = min(target_size[2], pos[2] + n/2 + 1)
    return float((x2-x1) * (y2-y1) * (z2-z1))


importanceMap = np.zeros(target_size, dtype=np.float32)
counts = 0.0
for k in filter_length:
    predictions = []
    imageIDs = {}
    with open(path_predictions + str(k) + '_fold_' + fold +  '_' + fold2 + '_full.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if int(row[2]) == -1 and int(row[3]) == -1 and int(row[4]) == -1:
                imageIDs[row[0]] = float(row[5])
            else:
                predictions.append([row[0], classes[row[1]], int(row[2]), int(row[3]), int(row[4]), float(row[5])])

    for i in range(len(predictions)):
        importance = predictions[i][5] - imageIDs[predictions[i][0]]
        if predictions[i][1] == classes['AD']:
            importance *= -1
        x = predictions[i][2]
        y = predictions[i][3]
        z = predictions[i][4]
        importanceMap[x, y, z] += importance / count_voxel(pos=(x,y,z), n=k)
    counts += len(imageIDs)


importanceMap /= counts

np.save('importanceMaps/importanceMap_2_135_fold_' + fold + '_' + fold2 + '_full.npy', importanceMap)

