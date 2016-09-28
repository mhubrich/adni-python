import numpy as np
from keras import backend as K
from scipy.stats.stats import pearsonr

from cnn.keras.cnn_codes.scans import load_scan, scan_to_array
from cnn.keras.models.vgg16.model import build_model
from utils.load_scans import load_scans
from utils.sort_scans import sort_groups

classes = ['AD', 'Normal']
path_ADNI = '/home/markus/Uni/Masterseminar/ADNI_gz/ADNI'
path_weights = '/home/markus/Uni/Masterseminar/adni/cnn/keras/vgg16/vgg16_weights_fc1_classes2.h5'

scans = load_scans(path_ADNI)
groups, _ = sort_groups(scans)

model = build_model(2)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.load_weights(path_weights)


def cnn_codes(nb_slice, nb_layer):
    codes = []
    layer_output = K.function([model.layers[0].input], [model.layers[nb_layer].output])
    i = 0
    for c in classes:
        codes.append([])
        for scan in groups[c]:
            x = load_scan(scan.path)
            x = scan_to_array(x, nb_slice, 'th')
            x = np.expand_dims(x, axis=0)
            output = layer_output([x])[0]
            tmp = []
            for o in output[0]:
                tmp.append(np.max(o))
            codes[i].append(tmp)
        i += 1
    return codes


def correlation(a, b):
    return pearsonr(a, b)[0]


def correlations(list1, list2, mode='same_class'):
    corr = []
    if mode == 'same_class':
        for i in xrange(0, len(list1)):
            for j in xrange(i+1, len(list2)):
                corr.append(correlation(list1[i], list2[j]))
    else:
        for i in xrange(0, len(list1)):
            for j in xrange(0, len(list2)):
                corr.append(correlation(list1[i], list2[j]))
    return corr


if __name__ == "__main__":
    slices = range(28, 68)
    layers = range(1, 24)
    info = 'Layer {layer:02d}, Slice {slice:02d}: [%.3f, %.3f, %.3f]'
    for l in layers:
        for s in slices:
            codes = cnn_codes(s, l)
            corr1 = correlations(codes[0], codes[0])
            corr2 = correlations(codes[1], codes[1])
            corr3 = correlations(codes[0], codes[1], mode='diff_class')
            finfo = info.format(layer=l, slice=s)
            print(finfo % (np.mean(corr1), np.mean(corr2), np.mean(corr3)))
