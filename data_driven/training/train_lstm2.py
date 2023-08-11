from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import layers
from keras.models import Model
import keras
import numpy as np
import h5py

LATENT_SIZE = 128
SIZE_PER_REYNOLD = 1000
NUMBER_OF_TIMESTEPS = 3
TIMESTEP_SIZE = 2
FILEPATH = "/workspace/velocityHistoryHuge.h5"
AUTOENCODER_PATH = "/workspace/models/autoencoder_2.hdf5"

# Load autoencoder and separate encoder and decoder
autoencoder = keras.models.load_model(AUTOENCODER_PATH)

# Encoder
encoder_input = autoencoder.input
encoder_output = autoencoder.layers[1].output
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)

# Decoder
decoder_input = autoencoder.layers[2].input
decoder_output = autoencoder.layers[-1].output
decoder_model = Model(inputs=decoder_input, outputs=decoder_output)

# Loading data and converting into latent representation using the encoder
with h5py.File(FILEPATH, 'r') as h5f:
    x_train = np.zeros((0, NUMBER_OF_TIMESTEPS, LATENT_SIZE))
    y_train = np.zeros((0, LATENT_SIZE))
    outputs_by_reynolds = {}
    for key in h5f.keys():
        data = h5f[key][:].T[:, 1:-1, 1:-1, :]
        data_latent = encoder_model.predict(data, verbose=1)
        del data
        number_of_groups = data_latent.shape[0] - (NUMBER_OF_TIMESTEPS * TIMESTEP_SIZE)
        inputs = np.zeros((number_of_groups, NUMBER_OF_TIMESTEPS, data_latent.shape[1]))
        outputs = np.zeros((number_of_groups, data_latent.shape[1]))
        for i in range(number_of_groups):
            upper_bound = i + (NUMBER_OF_TIMESTEPS * TIMESTEP_SIZE)
            inputs[i, :, :] = data_latent[i:upper_bound:TIMESTEP_SIZE, :]
            outputs[i, :] = data_latent[upper_bound, :]
        x_train = np.concatenate((x_train, inputs), axis=0)
        y_train = np.concatenate((y_train, outputs), axis=0)
        del data_latent

print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)

epochs = 500
batch_size = 32
activation_function_lstm = 'tanh'
activation_function_dense = 'linear'
number_of_lstm_cells = 43

lstm_model = keras.models.Sequential()
lstm_model.add(layers.LSTM(number_of_lstm_cells, activation=activation_function_lstm,
                            input_shape=(None, LATENT_SIZE), return_sequences=False))
lstm_model.add(layers.Dense(LATENT_SIZE, activation=activation_function_dense))
opt = Adam(learning_rate=1e-3)
lstm_model.compile(optimizer=opt, loss='mse')
history = lstm_model.fit(
    x_train,
    y_train,
    shuffle=True,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[ModelCheckpoint(filepath='/workspace/models/lstm_2.hdf5', save_best_only=True, monitor='loss', mode='min', verbose=1),
               ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='min', min_lr=1e-6)]
)
