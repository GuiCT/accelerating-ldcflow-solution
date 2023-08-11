# LEGACY, NOT USABLE
# run-autoencoder-discovery:
# 	docker run --rm --gpus all --name "autoencoder_discovery_gpu" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow:latest-gpu-jupyter /bin/bash -c "cd /workspace && pip install -r requirements.txt && python autoencoder_discovery/script.py"

# run-lstm-discovery:
# 	docker run --rm --name "lstm_discovery" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow /bin/bash -c "cd /workspace && pip install -r requirements.txt && python lstm_discovery/script.py"

# run-autoencoder-training:
# 	docker run --rm --gpus all --name "autoencoder_training" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow:latest-gpu-jupyter /bin/bash -c "cd /workspace && pip install -r requirements.txt && python training/train_autoencoder.py"

# run-lstm-training:
# 	docker run --rm --name "lstm_training" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow /bin/bash -c "cd /workspace && pip install -r requirements.txt && python train_lstm.py"

# run-autoencoder-discovery-2:
# 	docker run --rm --gpus all --name "autoencoder_discovery_gpu" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow:latest-gpu-jupyter /bin/bash -c "cd /workspace && pip install -r requirements.txt && python autoencoder_discovery2/script.py"

# run-lstm-discovery-2:
# 	docker run --rm --name "lstm_discovery" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow /bin/bash -c "cd /workspace && pip install -r requirements.txt && python lstm_discovery2/script.py"

# run-autoencoder2-training:
# 	docker run --rm --gpus all --name "autoencoder_training" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow:latest-gpu-jupyter /bin/bash -c "cd /workspace && pip install -r requirements.txt && python train_autoencoder2.py"

# run-lstm2-training:
# 	docker run --rm --name "lstm_training" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow /bin/bash -c "cd /workspace && pip install -r requirements.txt && python train_lstm2.py"

run-autoencoder-discovery:
	docker run --rm --gpus all --name "autoencoder_discovery_gpu" -v "$(shell pwd):/workspace" -it tensorflow/tensorflow:latest-gpu-jupyter /bin/bash -c "cd /workspace && pip install -r requirements.txt && for run in {1..10}; do python optuna_discovery/autoencoder/third_try/script.py; done"
