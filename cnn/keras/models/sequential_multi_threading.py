"""
Because Keras does not support pre-processing in parallel, we have to
implement a custom Sequential class containing two changes:
  (I) In build(), we use ModelMultiThreading instead of Model
  (II) In fit_generator(), we pass the argument 'nb_preprocessing_threads'
All other lines are copied from the original class:
https://github.com/fchollet/keras/blob/master/keras/models.py#L39
"""
import warnings

from keras.models import Sequential

from cnn.keras.models.model_multi_threading import ModelMultiThreading


class SequentialMultiThreading(Sequential):

    def build(self, input_shape=None):
        if not self.inputs or not self.outputs:
            raise Exception('Sequential model cannot be built: model is empty.'
                            ' Add some layers first.')
        # actually create the model
        self.model = ModelMultiThreading(self.inputs, self.outputs[0], name=self.name + '_model')

        # mirror model attributes
        self.supports_masking = self.model.supports_masking
        self._output_mask_cache = self.model._output_mask_cache
        self._output_tensor_cache = self.model._output_tensor_cache
        self._output_shape_cache = self.model._output_shape_cache
        self.input_layers = self.model.input_layers
        self.input_layers_node_indices = self.model.input_layers_node_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_node_indices = self.model.output_layers_node_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self.nodes_by_depth = self.model.nodes_by_depth
        self.container_nodes = self.model.container_nodes
        self.output_names = self.model.output_names
        self.input_names = self.model.input_names

        # make sure child model callbacks will call the parent Sequential model:
        self.model.callback_model = self

        self.built = True

    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight=None, max_q_size=10, nb_preprocessing_threads=4, **kwargs):
        if self.model is None:
            raise Exception('The model needs to be compiled before being used.')
        if 'show_accuracy' in kwargs:
            kwargs.pop('show_accuracy')
            warnings.warn('The "show_accuracy" argument is deprecated, '
                          'instead you should pass the "accuracy" metric to '
                          'the model at compile time:\n'
                          '`model.compile(optimizer, loss, '
                          'metrics=["accuracy"])`')
        if 'nb_worker' in kwargs:
            kwargs.pop('nb_worker')
            warnings.warn('The "nb_worker" argument is deprecated, '
                          'please remove it from your code.')
        if 'nb_val_worker' in kwargs:
            kwargs.pop('nb_val_worker')
            warnings.warn('The "nb_val_worker" argument is deprecated, '
                          'please remove it from your code.')
        if kwargs:
            raise Exception('Received unknown keyword arguments: ' +
                            str(kwargs))
        return self.model.fit_generator(generator,
                                        samples_per_epoch,
                                        nb_epoch,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=validation_data,
                                        nb_val_samples=nb_val_samples,
                                        class_weight=class_weight,
                                        max_q_size=max_q_size,
                                        nb_threads=nb_preprocessing_threads)
