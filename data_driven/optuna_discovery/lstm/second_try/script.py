# %% [markdown]
# # Extraindo dados gerados via método numérico
#
# Esses dados são armazenados em um arquivo HDF5 (Hierarchical Data Format 5)
#

# %%
from keras import layers
import optuna
import numpy as np
from keras.optimizers import Adam
from keras.layers import Dense, Reshape
from keras.models import Sequential, Model
import keras
import h5py

FILEPATH = '/workspace/velocityHistory.h5'
with h5py.File(FILEPATH, "r") as h5f:
    reynolds = {}
    for key in h5f.keys():
        reynolds[key] = h5f[key][:].T

# %% [markdown]
# ## Representando dados de treino e validação utilizando representação latente agrupados em 3 timesteps
#

# %% [markdown]
# ### Extraindo modelos de encoder e decoder do autoencoder utilizado
#
# Autoencoder 9 foi o que performou melhor, utilizando o mesmo.
#

# %%

original_dim = 63 * 63 * 2

autoencoder = keras.models.Sequential()
autoencoder.add(Reshape((original_dim,), input_shape=(63, 63, 2)))
autoencoder.add(Dense(64, activation='linear'))
autoencoder.add(Dense(original_dim, activation='selu'))
autoencoder.add(Reshape((63, 63, 2), input_shape=(original_dim,)))
opt = Adam(learning_rate=1e-5)
autoencoder.compile(optimizer=opt, loss='mse')

FILEPATH = '/workspace/models/autoencoder_2/9.h5'
autoencoder.load_weights(FILEPATH)

# Parte do encoder
encoder_input = autoencoder.input
encoder_output = autoencoder.layers[1].output
encoder_model = Model(inputs=encoder_input, outputs=encoder_output)
encoder_model.summary()

# Parte do decoder
decoder_input = autoencoder.layers[2].input
decoder_output = autoencoder.layers[-1].output
decoder_model = Model(inputs=decoder_input, outputs=decoder_output)
decoder_model.summary()

# %% [markdown]
# ## Transformando dados disponíveis na representação latente
#

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
#

# %%

LATENT_SIZE = 64

# Dados de treino: todos os Reynolds, com exceção de 3 aleatórios destinados para validação
validation_reynolds = np.random.choice(
    list(reynolds_latent.keys()), 3, replace=False)
x_train = np.zeros((0, 3, LATENT_SIZE))
y_train = np.zeros((0, LATENT_SIZE))
x_validation = np.zeros((0, 3, LATENT_SIZE))
y_validation = np.zeros((0, LATENT_SIZE))
for k, v in reynolds_latent.items():
    amount_of_groups = v.shape[0] - 3
    added_x = np.zeros((amount_of_groups, 3, LATENT_SIZE))
    added_y = np.zeros((amount_of_groups, LATENT_SIZE))
    for i in range(amount_of_groups):
        new_input = v[i:i+3].reshape(1, 3, LATENT_SIZE)
        new_output = v[i+3].reshape(1, LATENT_SIZE)
        added_x = np.concatenate((added_x, new_input), axis=0)
        added_y = np.concatenate((added_y, new_output), axis=0)
    if k in validation_reynolds:
        x_validation = np.concatenate((x_validation, added_x), axis=0)
        y_validation = np.concatenate((y_validation, added_y), axis=0)
    else:
        x_train = np.concatenate((x_train, added_x), axis=0)
        y_train = np.concatenate((y_train, added_y), axis=0)

# %% [markdown]
# # Gerando diferentes modelos de LSTM
#
# Hiperparâmetros alterados:
#
# - Função de ativação das células de LSTM
# - Função de ativação da camada de saída
# - Número de neurônios da camada de LSTM
#

# %%
# Hiperparâmetros fixos
epochs = 100
batch_size = 32
timesteps = 3

# %% [markdown]
# Procura de hiperparâmetros utilizando _framework_ Optuna
#

# %%


def objective(trial: optuna.Trial):
    activation_function_lstm = trial.suggest_categorical(
        'activation_function_lstm', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_dense = trial.suggest_categorical(
        'activation_function_dense', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    number_of_lstm_cells = trial.suggest_int('number_of_lstm_cells', 8, 256)

    lstm_network = keras.models.Sequential()
    lstm_network.add(layers.LSTM(number_of_lstm_cells, activation=activation_function_lstm,
                     input_shape=(None, 64), return_sequences=False))
    lstm_network.add(layers.Dense(64, activation=activation_function_dense))
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

    path = f'/workspace/models/lstm_2/{trial.number}.h5'
    lstm_network.save_weights(path)
    score = history.history['val_loss'][-1]
    # Checa se score é NaN
    # Se sim, retorna maxfloat para desmotivar uso de hiperparâmetros que causem esse comportamento
    if (np.isnan(score)):
        return np.finfo(np.float32).max, number_of_lstm_cells
    else:
        return score, number_of_lstm_cells


# %%
# Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
try:
    study = optuna.create_study(
        study_name='lstm_params_2', directions=['minimize', 'minimize'], storage='sqlite:////workspace/lstm_params_2.db')
except optuna.exceptions.DuplicatedStudyError:
    study = optuna.load_study(
        study_name='lstm_params_2', storage='sqlite:////workspace/lstm_params_2.db')

study.optimize(objective, n_trials=100, timeout=60 * 30)
exit(0)
