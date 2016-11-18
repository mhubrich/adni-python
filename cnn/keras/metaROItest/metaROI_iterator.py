# Iterator code copied from: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L419
import numpy as np

from keras.preprocessing.image import Iterator


class MetaROIIterator(Iterator):

    def __init__(self, numROI, class_indices, N, batch_size, shuffle, seed):
        self.k = numROI
        self.class_indices = class_indices
        super(MetaROIIterator, self).__init__(N, batch_size, shuffle, seed)

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.zeros((N, self.k), dtype=np.int32)
                for i in range(self.k):
                    index_array[:,i] = np.arange(N)
                if shuffle:
                    for c in range(len(self.class_indices)-1):
                        for i in range(self.k):
                            np.random.shuffle(index_array[self.class_indices[c]:self.class_indices[c+1], i])
                    np.random.shuffle(index_array)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

