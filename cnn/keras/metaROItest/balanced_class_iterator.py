# Iterator code copied from: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py#L419
import numpy as np

from keras.preprocessing.image import Iterator


class BalancedClassIterator(Iterator):

    def __init__(self, class_indices, N, batch_size, shuffle, seed):
        if shuffle:
            min_class = 999999
            for i in range(len(class_indices) - 1):
                if class_indices[i+1] - class_indices[i] < min_class:
                    min_class = class_indices[i+1] - class_indices[i]
            self.min_class = min_class
            self.class_indices = class_indices
            N = (len(class_indices) - 1) * min_class
            self.nb_sample = N
        super(BalancedClassIterator, self).__init__(N, batch_size, shuffle, seed)

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                if shuffle:
                    index_array = np.zeros(N, dtype=np.int32)
                    for i in range(len(self.class_indices) - 1):
                        index_array[i*self.min_class:(i+1)*self.min_class] = np.random.randint(low=self.class_indices[i],
                                                                                               high=self.class_indices[i+1],
                                                                                               size=self.min_class)
                    np.random.shuffle(index_array)
                else:
                    index_array = np.arange(N)

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

