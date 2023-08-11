# %% [markdown]
# # Extraindo dados gerados via método numérico
#
# Esses dados são armazenados em um arquivo HDF5 (Hierarchical Data Format 5)

# %%
import optuna
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras import layers
import keras
import h5py

FILEPATH = "/workspace/velocityHistory.h5"
with h5py.File(FILEPATH, "r") as h5f:
    reynolds = {}
    for key in h5f.keys():
        reynolds[key] = h5f[key][:].T

# %% [markdown]
# ## Representando dados de treino e validação utilizando representação latente agrupados em 3 timesteps

# %% [markdown]
# ### Extraindo modelos de encoder e decoder do autoencoder utilizado
#
# Autoencoder 19 foi o que performou melhor, utilizando o mesmo.
# Como o arquivo .hdf5 gerado só contém os pesos, é necessário compilar o modelo novamente para utilizá-lo.

# %%
# Hiperparâmetros do autoencoder 19
original_dim = 63 * 63 * 2
L1_dim = 31 * 31 * 2
L2_dim = 15 * 15 * 2
encoding_dim = 8 * 8 * 2
activation_function_encoder_l1 = 'selu'
activation_function_encoder_l2 = 'relu'
activation_function_encoder_l3 = 'relu'
activation_function_decoder_l1 = 'linear'
activation_function_decoder_l2 = 'relu'
activation_function_decoder_l3 = 'linear'

# %%

FILEPATH = '/workspace/models/autoencoder/19.hdf5'
autoencoder = keras.models.Sequential()
autoencoder.add(layers.Reshape((original_dim,), input_shape=(63, 63, 2)))
autoencoder.add(layers.Dense(
    L1_dim, activation=activation_function_encoder_l1))
autoencoder.add(layers.Dense(
    L2_dim, activation=activation_function_encoder_l2))
autoencoder.add(layers.Dense(
    encoding_dim, activation=activation_function_encoder_l3))
autoencoder.add(layers.Dense(
    L2_dim, activation=activation_function_decoder_l1))
autoencoder.add(layers.Dense(
    L1_dim, activation=activation_function_decoder_l2))
autoencoder.add(layers.Dense(
    original_dim, activation=activation_function_decoder_l3))
autoencoder.add(layers.Reshape((63, 63, 2), input_shape=(original_dim,)))
opt = Adam(learning_rate=1e-4)
autoencoder.compile(optimizer=opt, loss='mse')
# Carregando pesos
autoencoder.load_weights(FILEPATH)

# Parte do encoder
encoder_input = autoencoder.input
# Index 2 is the last layer of the encoder
encoder_output = autoencoder.layers[3].output
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)

# Parte do decoder
# Index 3 is the first layer of the decoder
decoder_input = autoencoder.layers[4].input
# Index -1 is the last layer of the decoder
decoder_output = autoencoder.layers[-1].output
decoder_model = Model(inputs=decoder_input, outputs=decoder_output)

# %% [markdown]
# ## Transformando dados disponíveis na representação latente

# %%
reynolds_latent = {}
for k, v in reynolds.items():
    v_inner = v[:, 1:-1, 1:-1, :]
    reynolds_latent[k] = encoder_model.predict(v_inner)

# %% [markdown]
# ## Agrupando valores latentes em grupos de 3 timesteps
#
# Input -> Conjunto de entrada com 3 valores sequenciais
#
# Output -> Quarto valor, se existir

# %%

x_train = np.zeros((0, 3, 128))
y_train = np.zeros((0, 128))
for v in reynolds_latent.values():
    amount_of_groups = v.shape[0] - 3
    for i in range(amount_of_groups):
        new_input = v[i:i+3].reshape(1, 3, 128)
        new_output = v[i+3].reshape(1, 128)
        x_train = np.concatenate((x_train, new_input), axis=0)
        y_train = np.concatenate((y_train, new_output), axis=0)


# %% [markdown]
# ## Separando 10% dos dados para validação
#
# Utiliza amostragem aleatória

# %%

permutation_idx = np.random.permutation(len(x_train))
x_train = x_train[permutation_idx]
y_train = y_train[permutation_idx]
ten_percent = int(len(x_train) * 0.1)
x_validation = x_train[:ten_percent]
y_validation = y_train[:ten_percent]
x_train = x_train[ten_percent:]
y_train = y_train[ten_percent:]

# %% [markdown]
# # Gerando diferentes modelos de LSTM
#
# Hiperparâmetros alterados:
# - Função de ativação das células de LSTM
# - Função de ativação da camada de saída
# - Número de neurônios da camada de LSTM

# %%
# Hiperparâmetros fixos
epochs = 100
batch_size = 32
timesteps = 3

# %% [markdown]
# Procura de hiperparâmetros utilizando _framework_ Optuna

# %%


def objective(trial: optuna.Trial):
    activation_function_lstm = trial.suggest_categorical(
        'activation_function_lstm', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_dense = trial.suggest_categorical(
        'activation_function_dense', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    number_of_lstm_cells = trial.suggest_int('number_of_lstm_cells', 8, 256)

    lstm_network = keras.models.Sequential()
    lstm_network.add(layers.LSTM(number_of_lstm_cells,
                     activation=activation_function_lstm, input_shape=(None, 128)))
    lstm_network.add(layers.Dense(128, activation=activation_function_dense))
    opt = Adam(learning_rate=1e-4)
    lstm_network.compile(optimizer=opt, loss='mse')
    try:
        history = lstm_network.fit(
            x_train,
            y_train,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_validation, y_validation),
        )
    except KeyboardInterrupt:
        trial.study.stop()

    path = f'/workspace/models/lstm/{trial.number}.hdf5'
    lstm_network.save_weights(path)
    score = history.history['val_loss'][-1]
    # Checa se score é NaN
    # Se sim, retorna maxfloat para desmotivar uso de hiperparâmetros que causem esse comportamento
    if (np.isnan(score)):
        return np.finfo(np.float32).max
    else:
        return score


# %%
# Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
try:
    study = optuna.create_study(
        study_name='lstm_params', directions=['minimize'], storage='sqlite:////workspace/lstm_params.db')
except optuna.exceptions.DuplicatedStudyError:
    study = optuna.load_study(
        study_name='lstm_params', storage='sqlite:////workspace/lstm_params.db')

study.optimize(objective, n_trials=5, timeout=60 * 30)
exit(0)
