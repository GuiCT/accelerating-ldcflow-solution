# Aceleração da solução computacional do Problema da Cavidade com Tampa Móvel

Estrutura dos diretórios:
- `julia`: implementações anteriores em Julia, atualmente movidas para o
diretório `LDCFlow`, que agrupa todos esses códigos em um *package*.
- `LDCFlow`: implementação atual em Julia, que forma o *package* `LDCFlow`. Esse
*package* pode ser adicionado ao *Registry* do Julia, e então utilizado em
qualquer lugar da máquina, desde que a pasta ainda exista.
- `matlab`: implementações originais na linguagem MATLAB, e as versões
refatoradas para melhor visualização.
- `referencia`: alguns resultados de simulação, serão refeitos.
- `resultados_cluster`: resultados de simulação obtidos no cluster LSNCS da
Unesp.
- `callbackexample.jl`: exemplo de como utilizar o *package* `LDCFlow` para
adicionar *callbacks* e salvar estados intermediários do algoritmo em disco.

Link para o artigo contendo a solução original do problema apresentado: https://www.fc.unesp.br/Home/Departamentos/Matematica/revistacqd2228/v17a18-solucao-numerica-da-equacao-de-poisson.pdf (código ISSN 2316-9664)