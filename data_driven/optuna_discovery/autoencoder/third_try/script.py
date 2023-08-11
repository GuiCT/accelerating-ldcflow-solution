# %% [markdown]
# # Localização da pasta do workspace

# %%
import optuna
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import layers
import keras
from optuna import Trial, TrialPruned
import h5py
import numpy as np
import sys
import os
import tensorflow as tf
# WORKSPACE_PATH = '/home/guilherme/Projects/accelerating-ldcflow-solution/data_driven'
WORKSPACE_PATH = '/workspace'

# %% [markdown]
# Carregando módulo utils

module_path = os.path.abspath(WORKSPACE_PATH)
print(module_path)
sys.path.insert(0, module_path)

from utils.file_loader import load_data
from utils.mean_max_error import MeanMaxError

# %% [markdown]
# # Extraindo dados gerados via método numérico
#
# Esses dados são armazenados em um arquivo HDF5 (Hierarchical Data Format 5)

# %%

CACHE_FILE_PATH = os.path.join(
    WORKSPACE_PATH, 'data', 'cached', 'trial_autoencoder_data.h5')
if not os.path.isfile(CACHE_FILE_PATH):
    data_file_path = os.path.join(
        WORKSPACE_PATH, 'data', 'autoencoder_data.h5')
    loaded_data = load_data(data_file_path)
    # Tomando uma amostra de 50% para as trials do Optuna
    # 10% desses 50% serão para validação, o resto será para treinamento durante as trials
    rng = np.random.default_rng()
    loaded_data = rng.permuted(loaded_data, axis=0)  # Realiza permutação
    trial_size = int(loaded_data.shape[0] * 0.5)
    loaded_data = np.resize(loaded_data, (trial_size,) + loaded_data.shape[1:])
    training_size = int(trial_size * 0.9)
    validation_size = int(trial_size * 0.1)
    training_data = np.copy(loaded_data[:training_size, :, :, :])
    validation_data = np.copy(loaded_data[training_size:, :, :, :])
    del loaded_data
    with h5py.File(CACHE_FILE_PATH, 'w') as h5f:
        h5f.create_dataset('training', data=training_data)
        h5f.create_dataset('validation', data=validation_data)
else:
    with h5py.File(CACHE_FILE_PATH, 'r') as h5f:
        training_data = h5f['training'][:]
        validation_data = h5f['validation'][:]

# %% [markdown]
# # Gerando diferentes modelos de Autoencoder
#
# Hiperparâmetros alterados:
# - Quantidade de camadas de encoder e decoder
# - Números de neurônios
# - Funções de ativação para cada layer

# %%
# Hiperparâmetros constantes
original_dim = 63 * 63 * 2
epochs = 50  # Deve ser o suficiente pra identificar as maiores discrepâncias

# %% [markdown]
# Procura de hiperparâmetros utilizando _framework_ Optuna

# %%


def objective(trial: Trial):
    batch_size = 32
    latent_size = trial.suggest_int('latent_size', 8, 512)
    # Latent layer
    activation_function_latent = trial.suggest_categorical(
        'activation_function_latent', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])

    # Intermediate layers
    n_intermediate_layers = trial.suggest_int('n_intermediate_layers', 0, 3)
    activation_functions_encoder = []
    activation_functions_decoder = []
    intermediate_layers_neurons = []
    for i in range(n_intermediate_layers):
        if i == 0:
            maximum = 63 * 63
        else:
            maximum = intermediate_layers_neurons[i - 1]
        intermediate_layers_neurons.append(trial.suggest_int(
            f'n_neurons_l{i + 1}', latent_size, maximum))
        activation_functions_encoder.append(trial.suggest_categorical(
            f'activation_function_encoder_l{i + 1}', ['selu', 'relu', 'linear', 'tanh', 'sigmoid']))
        activation_functions_decoder.append(trial.suggest_categorical(
            f'activation_function_decoder_l{i + 1}', ['selu', 'relu', 'linear', 'tanh', 'sigmoid']))

    # Last layer
    activation_function_last = trial.suggest_categorical(
        'activation_function_last', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])

    this_trial_model_file = os.path.join(
        WORKSPACE_PATH, 'models', 'autoencoder_3', f'{trial.number}.h5')
    autoencoder = keras.models.Sequential()
    autoencoder.add(layers.Reshape((original_dim,), input_shape=(63, 63, 2)))
    for i in range(n_intermediate_layers):
        autoencoder.add(layers.Dense(
            intermediate_layers_neurons[i], activation=activation_functions_encoder[i]))
    autoencoder.add(layers.Dense(
        latent_size, activation=activation_function_latent))
    for i in range(n_intermediate_layers - 1, -1, -1):
        autoencoder.add(layers.Dense(
            intermediate_layers_neurons[i], activation=activation_functions_decoder[i]))
    autoencoder.add(layers.Dense(
        original_dim, activation=activation_function_last))
    autoencoder.add(layers.Reshape((63, 63, 2), input_shape=(original_dim,)))
    opt = Adam(learning_rate=1e-5)
    earlystop_callback = EarlyStopping(
        monitor='val_loss', patience=5, min_delta=1e-5)
    loss_func = MeanMaxError()
    autoencoder.compile(optimizer=opt, loss=loss_func)
    print(autoencoder.summary())
    try:
        history = autoencoder.fit(
            training_data,
            training_data,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(validation_data, validation_data),
            callbacks=[earlystop_callback]
        )
    except KeyboardInterrupt:
        trial.study.stop()
    except Exception as e:
        print(f'Exception: {e}')
        raise TrialPruned()

    score = history.history['val_loss'][-1]
    autoencoder.save_weights(this_trial_model_file)
    # Checa se score é NaN
    # Se sim, retorna maxfloat para desmotivar uso de hiperparâmetros que causem esse comportamento

    if (np.isnan(score)):
        return np.finfo(np.float32).max
    else:
        return score


# %%

# Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
DATABASE_PATH = f"sqlite:///{os.path.join(WORKSPACE_PATH, 'optuna_discovery', 'databases', 'autoencoder_params_3.db')}"
try:
    study = optuna.create_study(
        study_name='autoencoder_params_3', directions=['minimize'], storage=DATABASE_PATH)
    study.enqueue_trial({"latent_size": 512})
    study.enqueue_trial({"latent_size": 256})
    study.enqueue_trial({"latent_size": 128})
except optuna.exceptions.DuplicatedStudyError:
    study = optuna.load_study(
        study_name='autoencoder_params_3', storage=DATABASE_PATH)

study.optimize(objective, n_trials=3, timeout=60 * 60)
exit(0)
