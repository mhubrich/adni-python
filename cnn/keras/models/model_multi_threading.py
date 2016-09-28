"""
Because Keras does not support pre-processing in parallel, we have to
implement a custom Model class containing two changes:
  (I) In fit_generator(), we pass the argument 'nb_threads'
  (II) In generator_queue(), we use nb_worker=nb_threads instead of nb_worker=1
All other lines are copied from the original class:
https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L461
"""
from keras.engine.training import Model, generator_queue, cbks
import warnings
import time


class ModelMultiThreading(Model):

    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight={}, max_q_size=10, nb_threads=4):  # We changed this
        wait_time = 0.01  # in seconds
        epoch = 0

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__'))
        if val_gen and not nb_val_samples:
            raise Exception('When using a generator for validation data, '
                            'you must specify a value for "nb_val_samples".')

        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks._set_model(callback_model)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': samples_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise Exception('validation_data should be a tuple '
                                '(val_x, val_y, val_sample_weight) '
                                'or (val_x, val_y). Found: ' + str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(val_x, val_y, val_sample_weight)
            self.validation_data = val_x + [val_y, val_sample_weights]
        else:
            self.validation_data = None

        # start generator thread storing batches into a queue
        data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size, nb_worker=nb_threads)  # We changed this

        callback_model.stop_training = False
        while epoch < nb_epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            while samples_seen < samples_per_epoch:
                generator_output = None
                while not _stop.is_set():
                    if not data_gen_queue.empty():
                        generator_output = data_gen_queue.get()
                        break
                    else:
                        time.sleep(wait_time)

                if not hasattr(generator_output, '__len__'):
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
                # build batch logs
                batch_logs = {}
                if type(x) is list:
                    batch_size = len(x[0])
                elif type(x) is dict:
                    batch_size = len(list(x.values())[0])
                else:
                    batch_size = len(x)
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                try:
                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)
                except Exception as e:
                    _stop.set()
                    raise

                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                batch_index += 1
                samples_seen += batch_size

                # epoch finished
                if samples_seen > samples_per_epoch:
                    warnings.warn('Epoch comprised more than '
                                  '`samples_per_epoch` samples, '
                                  'which might affect learning results. '
                                  'Set `samples_per_epoch` correctly '
                                  'to avoid this warning.')
                if samples_seen >= samples_per_epoch and do_validation:
                    if val_gen:
                        val_outs = self.evaluate_generator(validation_data,
                                                           nb_val_samples,
                                                           max_q_size=max_q_size)
                    else:
                        # no need for try/except because
                        # data has already been validated
                        val_outs = self.evaluate(val_x, val_y,
                                                 sample_weight=val_sample_weights,
                                                 verbose=0)
                    if type(val_outs) is not list:
                        val_outs = [val_outs]
                    # same labels assumed
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

        _stop.set()
        callbacks.on_train_end()
        return self.history
