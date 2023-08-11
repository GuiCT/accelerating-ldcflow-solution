from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import layers
from tqdm import tqdm
import keras
import numpy as np
import h5py
import os

# Workspace folder
WORKSPACE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This script path -> Parent folder -> data_generation/generated/autoencoder_data.h5
HDF5_FILEPATH = os.path.join(WORKSPACE_FOLDER, 'data_generation', 'generated', 'autoencoder_data.h5')
COMBINED_HDF5_PATH = os.path.join(WORKSPACE_FOLDER, 'data_generation', 'generated', 'combined_autoencoder_data.h5')
SAVED_MODEL_FILEPATH = os.path.join(WORKSPACE_FOLDER, 'models', 'autoencoder_x.hdf5')
input_shape = (63, 63, 2)
original_dim = np.prod(input_shape)
activation_latent = 'linear'
activation_last = 'selu'
batch_size = 32
epochs = 100

# Loading data in HDF5
# (n, 65, 65, 2)[] -> (m, 63, 63, 2)

if not os.path.exists(COMBINED_HDF5_PATH):
    print('Carregando dados contidos no arquivo HDF5')
    loaded_data = np.zeros((0, 63, 63, 2))
    with h5py.File(HDF5_FILEPATH, 'r') as h5f:
        for key in tqdm(h5f.keys()):
            dataset = h5f[key][:].T
            loaded_data = np.concatenate(
                [loaded_data, dataset[:, 1:-1, 1:-1, :]], axis=0)
    print('Dados carregados com sucesso')
    # Pre shuffling
    rng = np.random.default_rng()
    loaded_data = rng.permuted(loaded_data, axis=0)
    with h5py.File(COMBINED_HDF5_PATH, 'a') as h5f:
        # Separate in half
        half_size = loaded_data.shape[0] // 2
        h5f.create_dataset('first_half', data=loaded_data[:half_size, :, :, :])
        h5f.create_dataset('second_half', data=loaded_data[half_size:, :, :, :])
    del loaded_data


class HDF5DataLoader(Sequence):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        self.first_half = True
        self.load_dataset()
        self.num_samples = self.dataset.shape[0] * 2
        print(f'Tamanho do dataset combinado: {self.num_samples}')

    def load_dataset(self):
        with h5py.File(self.file_path, 'r') as h5f:
            if self.first_half:
                self.dataset = h5f['first_half'][:]
            else:
                self.dataset = h5f['second_half'][:]
            indices = np.arange(self.dataset.shape[0])
            np.random.shuffle(indices)
            self.dataset = self.dataset[indices]
            print(f'Loaded dataset, with shape: {self.dataset.shape}')

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):
        # Increment until end of first half
        # then load second half
        # when end of second half is reached
        # load first half again and go to the next epoch
        if idx == 0:
            self.first_half = True
            self.load_dataset()
        if not self.first_half:
            idx = idx - len(self) // 2 - 1
        lower_bound = idx * self.batch_size
        upper_bound = (idx + 1) * self.batch_size
        if upper_bound > self.dataset.shape[0]:
            self.first_half = not self.first_half
            ret_val = self.dataset[lower_bound:]
            self.load_dataset()
        else:
            ret_val = self.dataset[lower_bound:upper_bound]
        rng = np.random.default_rng()
        ret_val = rng.permuted(ret_val, axis=0)
        return ret_val, ret_val


autoencoder = keras.models.Sequential()
autoencoder.add(layers.Reshape((original_dim,), input_shape=input_shape))
autoencoder.add(layers.Dense(64, activation=activation_latent))
autoencoder.add(layers.Dense(original_dim, activation=activation_last))
autoencoder.add(layers.Reshape(input_shape))
opt = Adam(learning_rate=5e-4)
autoencoder.compile(optimizer=opt, loss='mse')
history = autoencoder.fit(
    HDF5DataLoader(COMBINED_HDF5_PATH, batch_size),
    shuffle=False,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[ModelCheckpoint(filepath=SAVED_MODEL_FILEPATH, save_best_only=True, monitor='loss', mode='min', verbose=1),
               ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='min', min_lr=1e-6)]
)
