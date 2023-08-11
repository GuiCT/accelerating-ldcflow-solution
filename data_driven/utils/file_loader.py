import h5py
import numpy as np
from tqdm import tqdm

def load_data(path: str):
    loaded_data = np.zeros((0, 63, 63, 2))
    with h5py.File(path, 'r') as h5f:
        print('Carregando dados de arquivo HDF5')
        for key in tqdm(h5f.keys()):
            dataset = h5f[key][:].T
            loaded_data = np.concatenate(
                [loaded_data, dataset[:, 1:-1, 1:-1, :]], axis=0)
    return loaded_data

