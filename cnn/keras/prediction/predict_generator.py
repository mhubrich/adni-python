from keras.engine.training import generator_queue
import time
import numpy as np


def predict_generator(model, generator, val_samples, max_q_size=10, nb_preprocessing_threads=4):
    """
    Generates predictions for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.
    # Arguments
        generator: generator yielding batches of input samples.
        val_samples: total number of samples to generate from `generator`
            before returning.
        max_q_size: maximum size for the generator queue
    # Returns
        Numpy array(s) of predictions.
    """
    model._make_predict_function()

    processed_samples = 0
    wait_time = 0.01
    all_outs = []
    filenames = []
    data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size, nb_worker=nb_preprocessing_threads)

    while processed_samples < val_samples:
        generator_output = None
        while not _stop.is_set():
            if not data_gen_queue.empty():
                generator_output = data_gen_queue.get()
                break
            else:
                time.sleep(wait_time)
        
        if isinstance(generator_output, tuple):
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
        else:
            x = generator_output
        try:
            outs = model.predict_on_batch(x)
        except:
            _stop.set()
            raise
        
        if type(x) is list:
            nb_samples = len(x[0])
        elif type(x) is dict:
            nb_samples = len(list(x.values())[0])
        else:
            nb_samples = len(x)

        if type(outs) != list:
            outs = [outs]

        if len(all_outs) == 0:
            for out in outs:
                shape = (val_samples,) + out.shape[1:]
                all_outs.append(np.zeros(shape))

        for i, out in enumerate(outs):
            try:
                all_outs[i][processed_samples:(processed_samples + nb_samples)] = out
            except ValueError:
                for k in range(0, val_samples-processed_samples):
                    all_outs[i][processed_samples + k] = out[k]

        processed_samples += nb_samples
        for f in y:
            filenames.append(f)
    
    _stop.set()
    if len(all_outs) == 1:
        return all_outs[0], filenames[0:val_samples]

    return all_outs, filenames[0:val_samples]
