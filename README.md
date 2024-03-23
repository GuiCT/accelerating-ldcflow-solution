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

---

Atualização (23/03/2024): dois trabalhos referente a esse repositório foram publicados no Congresso Nacional de Matemática Computacional e Aplicada de 2023, sendo eles:
- Resolução Numérica do Problema da Cavidade com Tampa Móvel utilizando a Linguagem de Programação Julia (https://proceedings.sbmac.org.br/sbmac/article/view/4161/4215)
  - Foca nos resultados obtidos pela redução no tempo de execução, isto é: novos resultados obtidos, para Números de Reynolds mais elevados do que nos artigos anteriores
- Otimização de um Algoritmo para Solução do Problema da Cavidade com Tampa Móvel (https://proceedings.sbmac.org.br/sbmac/article/view/4362)
  - Apresenta um breve resumo das diferenças nos tempos de execução ao realizar o _caching_ da fatoração utilizada para resolução do sistema linear esparso de Poisson

Somando-se à comparação de tempos de execução discutida no segundo trabalho, a apresentação realizada em ambas as fases do XXXV Congresso de Iniciação Científica da UNESP discute a melhoria obtida no tempo médio de iteração via uso do solucionador [PARDISO](https://www.intel.com/content/www/us/en/developer/articles/technical/pardiso-tips.html), presente na biblioteca oneMKL da Intel. É apresentada uma breve explicação de como multiprocessadores compartilham memória _cache_ e são apresentadas situações em que o tempo de preparação (_overhead_) é superior ao benefício obtido. Para uso do PARDISO, foi utilizado um [_wrapper_](https://github.com/JuliaSparse/Pardiso.jl) desenvolvido pela comunidade da Linguagem Julia para invocação desse _solver_. O poster da apresentação realizada pode ser encontrado em: https://eventos.reitoria.unesp.br/anais/xxxvcicunesp/771749-aceleracao-do-metodo-de-diferencas-finitas-para-o-problema-da-cavidade-com-tampa-movel-utilizando-linguagem-julia/
