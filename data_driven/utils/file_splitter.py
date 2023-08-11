import numpy as np
import h5py
from tqdm import tqdm
from file_loader import load_data

def split_h5_file(path: str, splitted_path: str):
    loaded_data = load_data(path)
    # Pre shuffling
    rng = np.random.default_rng()
    loaded_data = rng.permuted(loaded_data, axis=0)
    with h5py.File(splitted_path, 'a') as h5f:
        # Separate in half
        half_size = loaded_data.shape[0] // 2
        end_idx = half_size * 2 # Assert that both halfs are equal
        h5f.create_dataset('first_half', data=loaded_data[:half_size, :, :, :])
        h5f.create_dataset('second_half', data=loaded_data[half_size:end_idx, :, :, :])
    del loaded_data

