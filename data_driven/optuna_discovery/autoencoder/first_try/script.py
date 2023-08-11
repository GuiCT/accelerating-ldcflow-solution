# %% [markdown]
# # Extraindo dados gerados via método numérico
#
# Esses dados são armazenados em um arquivo HDF5 (Hierarchical Data Format 5)

# %%
from keras.optimizers import Adam
from keras import layers
import keras
import optuna
import numpy as np
import h5py

FILEPATH = "/workspace/velocityHistory.h5"
with h5py.File(FILEPATH, "r") as h5f:
    reynolds = {}
    for key in h5f.keys():
        reynolds[key] = h5f[key][:].T

# %% [markdown]
# # Gerando diferentes modelos de Autoencoder
#
# Hiperparâmetros alterados:
# - Funções de ativação para cada layer

# %%
# Hiperparâmetros constantes
original_dim = 63 * 63 * 2
L1_dim = 31 * 31 * 2
L2_dim = 15 * 15 * 2
encoding_dim = 8 * 8 * 2
epochs = 100

# %% [markdown]
# Preparando dados de treino do autoencoder
#
# Aprendizado não supervisionado, basta x_train

# %%

# Podemos agrupar diferentes Reynolds pois não há output no fit
reynolds_combined = np.concatenate(
    [reynolds[key] for key in reynolds.keys()], axis=0)
# Ignorando contorno pois é constante
reynolds_combined = reynolds_combined[:, 1:-1, 1:-1, :]

# %% [markdown]
# Procura de hiperparâmetros utilizando _framework_ Optuna

# %%


def objective(trial: optuna.Trial):
    batch_size = 32
    activation_function_encoder_l1 = trial.suggest_categorical(
        'activation_function_encoder_l1', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_encoder_l2 = trial.suggest_categorical(
        'activation_function_encoder_l2', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_encoder_l3 = trial.suggest_categorical(
        'activation_function_encoder_l3', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_decoder_l1 = trial.suggest_categorical(
        'activation_function_decoder_l1', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_decoder_l2 = trial.suggest_categorical(
        'activation_function_decoder_l2', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])
    activation_function_decoder_l3 = trial.suggest_categorical(
        'activation_function_decoder_l3', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])

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
    try:
        history = autoencoder.fit(
            reynolds_combined,
            reynolds_combined,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
        )
    except KeyboardInterrupt:
        trial.study.stop()

    path = f'/workspace/models/autoencoder/{trial.number}.hdf5'
    autoencoder.save_weights(path)
    score = history.history['loss'][-1]
    # Checa se score é NaN
    # Se sim, retorna maxfloat para desmotivar uso de hiperparâmetros que causem esse comportamento
    if (np.isnan(score)):
        return np.finfo(np.float32).max
    else:
        return score


# %%
# Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
if __name__ == "__main__":
    try:
        study = optuna.create_study(
            study_name='autoencoder_params', directions=['minimize'], storage='sqlite:////workspace/autoencoder_params.db')
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name='autoencoder_params', storage='sqlite:////workspace/autoencoder_params.db')

    study.optimize(objective, n_trials=7, timeout=60 * 30)
    exit(0)
