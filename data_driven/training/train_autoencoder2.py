from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import layers
import keras
import numpy as np
import h5py
import os

SIZE_PER_REYNOLD = 1000
FILEPATH = "/workspace/velocityHistoryHuge.h5"
TEMP_FILE_FILEPATH = "/workspace/velocityHistoryHugeTemp.h5"

# Create temp file if not exists
if not os.path.exists(TEMP_FILE_FILEPATH):
    with h5py.File(FILEPATH, "r") as h5f:
        with h5py.File(TEMP_FILE_FILEPATH, "a") as h5f_temp:
            half = SIZE_PER_REYNOLD // 2
            combined = np.zeros((0, 63, 63, 2))
            print('Salvando primeira metade do dataset')
            for key in h5f.keys():
                combined = np.concatenate(
                    [combined, h5f[key][:].T[:half, 1:-1, 1:-1, :]], axis=0)
            h5f_temp.create_dataset("first_half", data=combined)
            del combined
            combined = np.zeros((0, 63, 63, 2))
            print('Salvando segunda metade do dataset')
            for key in h5f.keys():
                combined = np.concatenate(
                    [combined, h5f[key][:].T[half:, 1:-1, 1:-1, :]], axis=0)
            h5f_temp.create_dataset("second_half", data=combined)
            del combined


original_dim = 63 * 63 * 2
encoding_dim = 8 * 8 * 2
epochs = 500
batch_size = 32
activation_function_latent = 'linear'
activation_function_last = 'selu'


class HDF5DataLoader(Sequence):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.first_half = True
        self.load_dataset()
        self.num_samples = self.dataset.shape[0] * 2

    def load_dataset(self):
        with h5py.File(self.file_path, 'r') as h5f:
            if self.first_half:
                self.dataset = h5f['first_half'][:]
            else:
                self.dataset = h5f['second_half'][:]
            indices = np.arange(self.dataset.shape[0])
            np.random.shuffle(indices)
            self.dataset = self.dataset[indices]

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):
        # Increment until end of first half
        # then load second half
        # when end of second half is reached
        # load first half again and go to the next epoch
        if not self.first_half:
            idx = idx - len(self) // 2
        lower_bound = idx * self.batch_size
        upper_bound = (idx + 1) * self.batch_size
        if upper_bound > self.dataset.shape[0]:
            self.first_half = not self.first_half
            ret_val = self.dataset[lower_bound:]
            self.load_dataset()
        else:
            ret_val = self.dataset[lower_bound:upper_bound]
        return ret_val, ret_val


autoencoder = keras.models.Sequential()
autoencoder.add(layers.Reshape((original_dim,), input_shape=(63, 63, 2)))
autoencoder.add(layers.Dense(
    encoding_dim, activation=activation_function_latent))
autoencoder.add(layers.Dense(
    original_dim, activation=activation_function_last))
autoencoder.add(layers.Reshape((63, 63, 2), input_shape=(original_dim,)))
opt = Adam(learning_rate=5e-4)
autoencoder.compile(optimizer=opt, loss='mse')
autoencoder.summary()
data_loader = HDF5DataLoader(TEMP_FILE_FILEPATH, batch_size)
history = autoencoder.fit(
    data_loader,
    shuffle=True,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[ModelCheckpoint(filepath='/workspace/models/autoencoder_2.hdf5', save_best_only=True, monitor='loss', mode='min', verbose=1),
               ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='min', min_lr=1e-6)]
)
