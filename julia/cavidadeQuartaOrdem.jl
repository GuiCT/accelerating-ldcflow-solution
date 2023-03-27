using SparseArrays;
using LinearAlgebra;

# Funções compartilhadas
include("cavidade_base.jl")

"""
    cavidadeQuartaOrdem(n::Int64, Re::Int64,
    δt::Float64, nt::Int64 = typemax(Int64), 
    xRange::Tuple{Float64, Float64}=(0.0, 1.0),
    yRange::Tuple{Float64, Float64}=(0.0, 1.0))

Resolve o problema da cavidade com tampa móvel utilizando aproximações de quarta ordem.

# Arguments
- `n::Int64`: número de elementos ao longo dos eixos x e y.
- `Re::Int64`: número de Reynolds.
- `δt::Float64`: passo de integração temporal.
- `nt::Int64`: quantidade máxima de iterações, por padrão o maior valor inteiro de 64 bits.
- `xRange::Tuple{Float64, Float64}`: intervalo no eixo x, por padrão de 0.0 à 1.0 (unitário).
- `yRange::Tuple{Float64, Float64}`: intervalo no eixo y, por padrão de 0.0 à 1.0 (unitário).

Os argumentos `nt`, `xRange` e `yRange` são opcionais.
"""
function cavidadeQuartaOrdem(
  n::Int64, Re::Int64,
  δt::Float64, nt::Int64=typemax(Int64),
  xRange::Tuple{Float64, Float64}=(0.0, 1.0),
  yRange::Tuple{Float64, Float64}=(0.0, 1.0)
)::Union{LDCFSolution, Nothing}
  # Inicializando matrizes e domínio do método
  x, y, δx, δy, ψ, u, v, ω = realizaAlocacoes(n, n, xRange, yRange)

  # Realizando a montagem da matriz de Poisson
  # E armazenando resultado da fatoração LDLT
  # A fatoração LDLT é utilizada na resolução do sistema linear esparso
  # Como a matriz é constante durante a execução do código, o resultado
  # Da fatoração LDLT também é. Sendo a tarefa mais intensiva do processo
  # de resolução do Sistema Linear, isso melhora de forma patente a performance
  # do código em questão.
  LDLT = ldlt(matrizPoisson(n))

  for iterationNumber in 1:nt
    ω = calculoContornoω!(δx, δy, ψ, ω)
    ω = atualizandoω(Re, δx, δy, δt, ω, u, v)
    # Atualizando ψ a partir do sistema linear esparso de Poisson
    ψ[2:n, 2:n] .= resolucaoSistemaLinear(δx, -ω, LDLT)
    u₀ = copy(u)
    v₀ = copy(v)
    u, v = atualizandoUVQuartaOrdem!(δx, δy, ψ, u, v)

    # Calculando resíduos em u e v
    residuoU = maximum(abs.(u - u₀))
    residuoV = maximum(abs.(v - v₀))
    # Printando informações do passo
    println("Passo: ", iterationNumber, "\tResíduo (u): ", residuoU, "\tResíduo (v): ", residuoV)

    # Se um dos erros for maior que 1e+8, aborta.
    # Se o erro de ambos forem menores que 1e-5, logo, convergiu.
    if (residuoU > 1e+8 || residuoV > 1e+8)
      println("Erro maior que 1e+8, abortando...")
      return Nothing
    elseif (residuoU < 1e-5 && residuoV < 1e-5)
      println("Convergiu!")
      return LDCFSolution(
        x, y, u, v, true
      )
    end
  end

  return LDCFSolution(
    x, y, u, v, false
  )
end


function geraVetorDePadrao(
  size::Int, default::Number,
  value::Number, each::Int,
  offset::Int = 0)::Vector{Float64}

  vector = fill(default, size)
  for i ∈ (each + offset):each:size
    vector[i] = value
  end

  return vector
end

function matrizPoisson(n::Int)
  # Inicializando matriz esparsa
  return spdiagm(
    - n => geraVetorDePadrao( # [1, 1, ..., 0, 1, 1, ...]
        (n - 2)  * (n - 1) - 1,
        1.0,
        0.0,
        n - 1
      ),
    - (n - 1) => fill(4.0, (n - 2) * (n - 1)), # [4, 4, ..., 4]
    - (n - 2) => geraVetorDePadrao( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2)  * (n - 1) + 1,
      1.0,
      0.0,
      n - 1,
      1
    ),
    -1 => geraVetorDePadrao( # [4, 4, ..., 0, 4, 4, ...]
      (n - 1)^2 - 1,
      4.0,
      0.0,
      n - 1
    ),
    0 => fill(-20.0, (n - 1)^2), # [-20, -20, ..., -20]
    1 => geraVetorDePadrao( # [4, 4, ..., 0, 4, 4, ...]
      (n - 1)^2 - 1,
      4.0,
      0.0,
      n - 1
    ),
    n - 2 => geraVetorDePadrao( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2)  * (n - 1) + 1,
      1.0,
      0.0,
      n - 1,
      1
    ),
    n - 1 => fill(4.0, (n - 2) * (n - 1)), # [4, 4, ..., 4]
    n => geraVetorDePadrao( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2)  * (n - 1) - 1,
      1.0,
      0.0,
      n - 1
    )
  )
end

function resolucaoSistemaLinear(δ::Float64, f::Matrix{Float64}, LDLT)
  sizeTotal = size(f)[1]
  sizeInterno = sizeTotal - 2
  R = zeros(sizeInterno^2)

  @inbounds Threads.@threads for j ∈ 1:sizeInterno
    for i ∈ 1:sizeInterno
      # T[i, j] -> R[(i - 1) * sizeInterno + j] (2D -> 1D)
      R[(i - 1) * sizeInterno + j] = δ^2 * (
        f[i+2, j+1] +
        f[i+1, j] +
        8 * f[i+1, j+1] +
        f[i, j+1] +
        f[i+1, j+2]
      ) / 2
    end
  end

  # Obtém vetor resultante da resolução do sistema linear
  # Ax = R -> x = A\R
  # Utiliza Fatoração LDLT de A
  x = LDLT \ R # Resolve o sistema linear

  # Retorna o vetor x no formato de matriz
  # Função reshape não realiza novas alocações
  return transpose(reshape(x, (sizeInterno, sizeInterno)))
end
