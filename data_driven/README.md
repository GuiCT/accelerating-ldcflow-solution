# Abordagem _data-driven_ para resolução do problema

Nesse diretório, a ideia é executar o algoritmo original de quarta ordem, armazenar os resultados utilizados para diferentes números de Reynolds, e a partir desses dados, treinar uma rede neural _Autoencoder_, utilizada para reduzir a dimensionalidade do problema em questão. Após isso, a representação latente gerada por essa rede será utilizada como entrada para uma rede neural _Long Short-Term Memory_ (LSTM), que será treinada para prever o próximo passo da solução. Após obter o próximo passo da solução, o _Autoencoder_ será utilizado para reconstruir a solução completa, e o processo se repete, até atingir convergência.

Trata-se de uma nova abordagem para resolução do problema, uma investigação da viabilidade de aplicar essa abordagem nesse problema em questão.

# Diretórios

- `data`: dados gerados via execução de método numérico, isto é, tensores de tamanho (63, 63, 2) indicando as componentes da velocidade em cada ponto da malha, excluído o contorno.
- `models`: modelos gerados pelas _trials_ do Optuna e modelos escolhidos a partir dessas para um treinamento "definitivo", com maior conjunto de dados e _epochs_.
- `optuna_discovery`: diretório destinado para procura de hiperparâmetros utilizando a _framework_ Optuna
- `training`: diretório contendo _scripts_ utilizados para treinar as redes "definitivas"
- `utils`: diretório contendo utilitários utilizados em outros _scripts_ que utilizam a Linguagem Python
- `Makefile`: arquivo utilizado para armazenar _scripts_ de linha de comando utilizados muitas vezes
- `requirements.txt`: arquivo contendo todas as bibliotecas de Python necessárias para execução dos _scripts_ nesta pasta

# Conclusão

11/08/2023: decidi que vou deixar o estudo dessa abordagem de lado **por enquanto**. Dentre as dificuldades encontradas estão:
1. Montagem de conjunto de dados para treino/validação
    1. Os métodos numéricos utilizados tendem a apresentar um estado turbulento/instável ao início de sua execução, mas a partir de determinada iteração, apresentam estabilidade. Dessa forma, é necessário formular uma abordagem que não cause a montagem de um conjunto de dados muito redundante e desbalanceado.
    2. O alto volume de dados gerado necessita de uma alta capacidade de memória primária (RAM) e muitas vezes não pode ser persistido na mesma, obrigando o computador à fazer muitos acessos à memória secundária, cuja velocidade é muito menor. Isso acaba por impactar o tempo de treinamento, validação e teste das redes neurais aqui estudadas.
2. Alto tempo de execução
    1. Todos os testes aqui demonstrados foram realizados em minha máquina pessoal. Embora esses processos sejam altamente automatizáveis, os mesmos tomam muito tempo de processamento, impossibilitando o uso da máquina para outros propósitos durante esse período.
3. Alto consumo de recursos computacionais
    1. Análogo ao que foi apontado em 2.1, o alto uso de recursos como GPU e CPU impossibilitam qualquer outro uso enquanto o _Optuna_ está realizando _trials_ ou uma rede escolhida a partir destas está sendo treinada.

As duas primeiras tentativas de formulação da estratégia Autoencoder + LSTM (cujos parâmetros encontrados via Optuna foram mantidos nos bancos de dados localizados em `optuna_discovery/databases`) pareciam promissoras, no entanto, isso se deu pois todo o treinamento foi feito em cima do conjunto desbalanceado, como foi anotado em 1.1. Dessa forma, quando os modelos encontrados e treinados usando esse conjunto de dados foi aplicado para resolução do problema em si, seu desempenho foi pífio, apresentando erro percentual de 50% em um intervalo de 1000 iterações.

# O quê mudar no futuro

Para reavaliar essa possibilidade de aplicação no futuro, ficam algumas lições:

- Tomar cuidado com balanceamento do conjunto de dados
- Utilizar arquitetura adequada para execução das _trials_
- Talvez utilizar uma tecnologia mais evoluída. Hoje, a plataforma do _PyTorch_ é muito mais atrativa do que a combinação _Keras_ + _Tensorflow_