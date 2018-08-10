"""
super important if you wish to train on multi-gpus# keras has a bug that can't save multi-gpu model
because it's scatter into N pieces(depending on count of GPUs)
So all you need to do is saving the model which haven't been scattered.

Use it almost like ModelCheckpoint

For example:
    Single-GPU:
        checkpoints = ModelCheckpoint(filepath=fname, monitor="val_loss", mode="min",
                                     save_best_only=True, verbose=1)
    Multi-GPU:
        checkpoints = ParallelModelCheckpoint(single_gpu_model, filepath=fname, monitor="val_loss", mode="min",
                                         save_best_only=True, save_weights_only=False, verbose=1)
"""
import os

import warnings
from keras.callbacks import ModelCheckpoint


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, single_gpu_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):

        name = 'weights' if save_weights_only else 'model'

        if not save_best_only:
            name += '-{epoch:03d}-{val_loss:.4f}.hdf5'
            fname = os.path.sep.join([filepath, name])
        else:
            name += '_best.hdf5'
            fname = os.path.sep.join([filepath, name])

        super(ParallelModelCheckpoint, self).__init__(fname, monitor, verbose,
                                                      save_best_only, save_weights_only,
                                                      mode, period)

        # hardcore
        # hardcore
        # hardcore
        self.model_to_save = single_gpu_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model_to_save.save_weights(filepath, overwrite=True)
                        else:
                            self.model_to_save.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model_to_save.save_weights(filepath, overwrite=True)
                else:
                    self.model_to_save.save(filepath, overwrite=True)