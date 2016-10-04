from keras import backend as K
from keras.layers.pooling import MaxPooling3D


class MinPooling3D(MaxPooling3D):
    '''Min pooling operation for 3D data (spatial or spatio-temporal).
    # Arguments
        pool_size: tuple of 3 integers,
            factors by which to downscale (dim1, dim2, dim3).
            (2, 2, 2) will halve the size of the 3D input in each dimension.
        strides: tuple of 3 integers, or None. Strides values.
        border_mode: 'valid' or 'same'.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 4.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
    # Input shape
        5D tensor with shape:
        `(samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, len_pool_dim1, len_pool_dim2, len_pool_dim3, channels)` if dim_ordering='tf'.
    # Output shape
        5D tensor with shape:
        `(nb_samples, channels, pooled_dim1, pooled_dim2, pooled_dim3)` if dim_ordering='th'
        or 5D tensor with shape:
        `(samples, pooled_dim1, pooled_dim2, pooled_dim3, channels)` if dim_ordering='tf'.
    '''

    def __init__(self, pool_size=(2, 2, 2), strides=None, border_mode='valid',
                 dim_ordering='default', **kwargs):
        super(MinPooling3D, self).__init__(pool_size, strides, border_mode,
                                           dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          border_mode, dim_ordering):
        output = -K.pool3d(-inputs, pool_size, strides,
                          border_mode, dim_ordering, pool_mode='max')
        return output

