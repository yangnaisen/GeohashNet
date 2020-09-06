import keras
import os


class CheckPointer(keras.callbacks.Callback):
    def __init__(self,cpu_model,checkpoint_dir):
        super(CheckPointer, self).__init__()
        self.cpu_model = cpu_model
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs={}):
        num_epoch = epoch+1
        weights_name = f'epoch_{num_epoch:04d}_weights.h5py'
        file_path = os.path.join(self.checkpoint_dir,weights_name)
        self.cpu_model.save_weights(file_path)

