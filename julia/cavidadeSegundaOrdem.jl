using SparseArrays;
using LinearAlgebra;

# Funções compartilhadas
include("cavidade_base.jl")

"""
    cavidadeSegundaOrdem(nx::Int64, ny::Int64, Re::Int64,
    δt::Float64, nt::Int64 = typemax(Int64), 
    xRange::Tuple{Float64, Float64}=(0.0, 1.0),
    yRange::Tuple{Float64, Float64}=(0.0, 1.0))

Resolve o problema da cavidade com tampa móvel utilizando aproximações de segunda ordem.

# Arguments
- `nx::Int64`: número de elementos ao longo do eixo x.
- `ny::Int64`: número de elementos ao longo do eixo y.
- `Re::Int64`: número de Reynolds.
- `δt::Float64`: passo de integração temporal.
- `nt::Int64`: quantidade máxima de iterações, por padrão o maior valor inteiro de 64 bits.
- `xRange::Tuple{Float64, Float64}`: intervalo no eixo x, por padrão de 0.0 à 1.0 (unitário).
- `yRange::Tuple{Float64, Float64}`: intervalo no eixo y, por padrão de 0.0 à 1.0 (unitário).

Os argumentos `nt`, `xRange` e `yRange` são opcionais.
"""
function cavidadeSegundaOrdem(
  nx::Int64, ny::Int64, Re::Int64,
  δt::Float64, nt::Int64=typemax(Int64),
  xRange::Tuple{Float64, Float64}=(0.0, 1.0),
  yRange::Tuple{Float64, Float64}=(0.0, 1.0);
  atol::Float64=1e-6, rtol::Float64=1e-5, limit::Float64=1e+8
)::Union{LDCFSolution, Nothing}
  # Inicializando matrizes e domínio do método
  x, y, δx, δy, ψ, u, v, ω = realizaAlocacoes(nx, ny, xRange, yRange)

  # Realizando a montagem da matriz de Poisson
  # E armazenando resultado da fatoração LU
  # A fatoração LU é utilizada na resolução do sistema linear esparso
  # Como a matriz é constante durante a execução do código, o resultado
  # Da fatoração LU também é. Sendo a tarefa mais intensiva do processo
  # de resolução do Sistema Linear, isso melhora de forma patente a performance
  # do código em questão.
  LU = lu(matrizPoissonSegundaOrdem(nx, ny, δx, δy))

  for iterationNumber in 1:nt
    ω = calculoContornoω!(δx, δy, ψ, ω)
    ω = atualizandoω(Re, δx, δy, δt, ω, u, v)
    # Atualizando ψ a partir do sistema linear esparso de Poisson
    ψ .= resolucaoSistemaLinear(ω, LU)
    u₀ = copy(u)
    v₀ = copy(v)
    u, v = atualizandoUVSegundaOrdem!(δx, δy, ψ, u, v)

    residuoU = maximum(abs.(u - u₀))
    residuoV = maximum(abs.(v - v₀))
    # Printando informações do passo
    println("Passo: ", iterationNumber, "\tResíduo (u): ", residuoU, "\tResíduo (v): ", residuoV)

    if (residuoU > limit || residuoV > limit)
      println("Falhou em atingir convergência.")
      return Nothing
    elseif (residuoU < rtol && residuoV < rtol)
      println("Convergiu dentro da tolerância especificada.")
      return LDCFSolution(
        x, y, u, v, Re, true
      )
    end
  end

  return LDCFSolution(
    x, y, u, v, Re, false
  )
end

function matrizPoissonSegundaOrdem(nx::Int64, ny::Int64, δx::Float64, δy::Float64)
  quantidadeElementos = 5 * (nx * ny + 1) - 3 * (nx + ny)
  # Vetores contendo posições de linha, coluna e o elemento em si
  indicesLinha = zeros(Int64, quantidadeElementos)
  indicesColuna = zeros(Int64, quantidadeElementos)
  elementos = zeros(Float64, quantidadeElementos)
  idx = 1

  for i in 1:nx+1
    for j in 1:ny+1
      flatIndex = (i - 1) * (ny + 1) + j
      # Contorno -> Preenche diagonal principal com 1 (identidade)
      if (i == 1) || (i == nx + 1) || (j == 1) || (j == ny + 1)
        indicesLinha[idx] = flatIndex
        indicesColuna[idx] = flatIndex
        elementos[idx] = 1
        idx += 1
        # Fora do contorno -> Matriz pentadiagonal
      else
        # Elemento da esquerda
        indicesLinha[idx] = flatIndex
        indicesColuna[idx] = (i-2)*(ny+1)+j
        elementos[idx] = δx^-2
        idx += 1

        # Elemento do centro
        indicesLinha[idx] = flatIndex
        indicesColuna[idx] = (i-1)*(ny+1)+j
        elementos[idx] = -2 * (δx^-2 + δy^-2)
        idx += 1

        # Elemento da direita
        indicesLinha[idx] = flatIndex
        indicesColuna[idx] = (i)*(ny+1)+j
        elementos[idx] = δx^-2
        idx += 1

        # Elemento de baixo
        indicesLinha[idx] = flatIndex
        indicesColuna[idx] = (i-1)*(ny+1)+j-1
        elementos[idx] = δy^-2
        idx += 1

        # Elemento de cima
        indicesLinha[idx] = flatIndex
        indicesColuna[idx] = (i-1)*(ny+1)+j+1
        elementos[idx] = δy^-2
        idx += 1
      end
    end
  end

  return sparse(indicesLinha, indicesColuna, elementos)
  # return A
end

function resolucaoSistemaLinear(ω::Matrix{Float64}, LU)  
  nx, ny = size(ω)
  b = zeros(nx * ny)

  # Formando vetor independente b a partir de ω
  # No contorno -> 0.0
  # Internamente -> -ω
  @inbounds Threads.@threads for j ∈ 2:ny-1
    for i ∈ 2:nx-1
      b[(i - 1) * ny + j] = - ω[i, j]
    end
  end

  # Obtém vetor resultante da resolução do sistema linear
  # Ax = b -> x = A\b
  # Utiliza Fatoração LU de A
  x = LU \ b

  # Retorna o vetor x no formato de matriz
  # Função reshape não realiza novas alocações
  return transpose(reshape(x, (ny, nx)))
end
