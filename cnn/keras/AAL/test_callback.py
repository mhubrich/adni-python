from cnn.keras.evaluation_callback import Evaluation


class Test(Evaluation):
    def __init__(self, generator, path, callbacks=[]):
        super(Test, self).__init__(generator, callbacks=callbacks)
        self.path = path
        self.monitor = ['val_loss', 'val_acc', 'val_fmeasure', 'val_mcc', 'val_mean_acc']

    def on_epoch_end(self, epoch, logs={}):
        d = {}
        super(Test, self).on_epoch_end(epoch, d)
        l = '0s - loss: %.4f - acc: %.4f' % (logs['loss'], logs['acc'])
        for m in self.monitor:
            l += ' - ' + m + (': %.4f' % d[m])
        l += '\n'
        with open(self.path, 'a') as myfile:
            myfile.write(l)

