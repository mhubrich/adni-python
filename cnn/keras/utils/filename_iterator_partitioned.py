import numpy as np

from cnn.keras.utils.filename_iterator import FilenameIterator


class FilenameIteratorPartitioned(FilenameIterator):

    def len_grid(self):
        return len(self.grid[0])

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = [np.zeros((current_batch_size,) + self.image_shape) for _ in range(len(self.grid))]
        batch_y = []
        # build batch of image data
        for i, j in enumerate(index_array):
            j_scan, j_voxel = divmod(j, self.len_grid())
            if self.load_all_scans:
                scan = self.scans[j_scan]
            else:
                scan = self.load_scan(self.scans[j_scan])
            for k in range(len(self.grid)):
                voxel = self.grid[k][j_voxel]
                x = self.get_scan(scan=scan, voxel=voxel, target_size=self.target_size)
                # x = self.image_data_generator.standardize(x)
                x = self.expand_dims(x, self.dim_ordering)
                batch_x[k][i] = x
            batch_y.append(self.get_voxel_str(self.grid[0][j_voxel]) + '_' + self.filenames[j_scan])
        return batch_x, batch_y
