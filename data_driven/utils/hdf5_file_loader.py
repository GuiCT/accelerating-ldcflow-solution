import numpy as np
from keras.utils import Sequence
import h5py

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

