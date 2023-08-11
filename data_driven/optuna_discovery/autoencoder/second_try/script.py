# %% [markdown]
# # Extraindo dados gerados via método numérico
#
# Esses dados são armazenados em um arquivo HDF5 (Hierarchical Data Format 5)

# %%
from keras.callbacks import ModelCheckpoint
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
# - Quantidade de camadas de encoder e decoder
# - Números de neurônios
# - Funções de ativação para cada layer

# %%
# Hiperparâmetros constantes
original_dim = 63 * 63 * 2
epochs = 66

# %% [markdown]
# Preparando dados de treino do autoencoder
#
# Aprendizado não supervisionado, basta x_train

# %%

# Selecionando 3 Reynolds aleatórios para validação
reynolds_validation = np.random.choice(list(reynolds.keys()), 3)
validation_data = np.concatenate(
    [reynolds[key][:, 1:-1, 1:-1, :] for key in reynolds_validation], axis=0)
training_data = np.concatenate([reynolds[key][:, 1:-1, 1:-1, :]
                               for key in reynolds.keys() if key not in reynolds_validation], axis=0)

# %% [markdown]
# Procura de hiperparâmetros utilizando _framework_ Optuna

# %%


def objective(trial: optuna.Trial):
    batch_size = 32
    latent_size = trial.suggest_int('latent_size', 8, 64, log=True)
    # Latent layer
    activation_function_latent = trial.suggest_categorical(
        'activation_function_latent', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])

    # Intermediate layers
    n_intermediate_layers = trial.suggest_int('n_intermediate_layers', 0, 2)
    activation_functions_encoder = []
    activation_functions_decoder = []
    intermediate_layers_neurons = []
    for i in range(n_intermediate_layers):
        if i == 0:
            maximum = 63 * 63
        else:
            maximum = intermediate_layers_neurons[i - 1]
        intermediate_layers_neurons.append(trial.suggest_int(
            f'n_neurons_l{i + 1}', latent_size, maximum, log=True))
        activation_functions_encoder.append(trial.suggest_categorical(
            f'activation_function_encoder_l{i + 1}', ['selu', 'relu', 'linear', 'tanh', 'sigmoid']))
        activation_functions_decoder.append(trial.suggest_categorical(
            f'activation_function_decoder_l{i + 1}', ['selu', 'relu', 'linear', 'tanh', 'sigmoid']))

    # Last layer
    activation_function_last = trial.suggest_categorical(
        'activation_function_last', ['selu', 'relu', 'linear', 'tanh', 'sigmoid'])

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
    autoencoder.compile(optimizer=opt, loss='mse')
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
        )
    except KeyboardInterrupt:
        trial.study.stop()

    score = history.history['val_loss'][-1]
    autoencoder.save_weights(
        f'/workspace/models/autoencoder_2/{trial.number}.h5')
    # Checa se score é NaN
    # Se sim, retorna maxfloat para desmotivar uso de hiperparâmetros que causem esse comportamento

    # Utiliza número de camadas intermediárias como segunda métrica

    if (np.isnan(score)):
        return np.finfo(np.float32).max, (n_intermediate_layers - 1) * 2
    else:
        return score, (n_intermediate_layers - 1) * 2


# %%
# Se o estudo já existe no banco de dados, carrega-o, caso contrário, cria um novo
try:
    study = optuna.create_study(
        study_name='autoencoder_params_2', directions=['minimize', 'minimize'], storage='sqlite:////workspace/autoencoder_params_2.db')
except optuna.exceptions.DuplicatedStudyError:
    study = optuna.load_study(
        study_name='autoencoder_params_2', storage='sqlite:////workspace/autoencoder_params_2.db')

study.optimize(objective, n_trials=21, timeout=60 * 20)
exit(0)
