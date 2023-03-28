# Verificando número de iterações do Método Multigrid para diferentes Fatores de Relaxação

Durante as primeiras 1000 iterações do Método da Cavidade, foi armazenada a quantidade de iterações do Método de Multigrid para cada iteração do Método da Cavidade. Esse teste foi realizado com os fatores de relaxação 0.1, 0.2, 0.3, ..., 2.0, as malhas n=32, n=64 e n=128, com Re=100, Re=400 e dt=0.001

Os resultados podem ser observados nos arquivos .csv no formato iterationsMG_{n}_Re{Re}.csv, onde cada linha representa um fator de relaxação. Quando a linha apresenta um único número 0, isso quer dizer que o Método Multigrid falhou em atingir convergência.