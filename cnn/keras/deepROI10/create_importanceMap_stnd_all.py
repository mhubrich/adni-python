import numpy as np
import csv


target_size = (22, 22, 22)
classes = {'Normal':0, 'AD':1}


def count_voxel(pos=(0,0,0), n=3):
    x1 = max(0, pos[0] - n/2)
    x2 = min(target_size[0], pos[0] + n/2 + 1)
    y1 = max(0, pos[1] - n/2)
    y2 = min(target_size[1], pos[1] + n/2 + 1)
    z1 = max(0, pos[2] - n/2)
    z2 = min(target_size[2], pos[2] + n/2 + 1)
    return float((x2-x1) * (y2-y1) * (z2-z1))


def create_importanceMap(filter_length, fold, path):
    importanceMap = {}
    for k in filter_length:
        predictions = []
        imageIDs = {}
        with open(path + str(k) + '_' + fold + '_stnd_all.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if int(row[2]) == -1 and int(row[3]) == -1 and int(row[4]) == -1:
                    imageIDs[row[0]] = float(row[5])
                else:
                    predictions.append([row[0], classes[row[1]], int(row[2]), int(row[3]), int(row[4]), float(row[5])])

        for i in range(len(predictions)):
            if predictions[i][0] not in importanceMap:
                importanceMap[predictions[i][0]] = np.zeros(target_size, dtype=np.float32)
            importance = predictions[i][5] - imageIDs[predictions[i][0]]
            if predictions[i][1] == classes['AD']:
                importance *= -1
            x = predictions[i][2]
            y = predictions[i][3]
            z = predictions[i][4]
            importanceMap[predictions[i][0]][x, y, z] += importance / count_voxel(pos=(x,y,z), n=k)

    for key in importanceMap:
        importanceMap[key] /= len(filter_length)

    return importanceMap

