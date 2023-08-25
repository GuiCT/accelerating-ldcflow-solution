using FiniteDifferences;
using SparseArrays;
using LinearAlgebra;

"""
Malha linear
- nx: Número de pontos na coordenada x
- ny: Número de pontos na coordenada y
- δx: Espaçamento entre pontos na coordenada x
- δy: Espaçamento entre pontos na coordenada y
"""
struct LinearMesh
  nx::Int64
  ny::Int64
  δx::Float64
  δy::Float64
end

"""
Resultado obtido
- mesh: Coordenadas dos pontos no espaço
- V: Velocidade em cada ponto do domínio
"""
struct LDCFSolution
  mesh::LinearMesh
  V::Array{Float64,3}
end

"""
Dados sobre a execução do método
- code: Motivo de parada do método
- method: Método utilizado
- numberOfIterations: Número de iterações realizadas
- overheadTime: Tempo de preparação do método (s)
- executionTime: Tempo de execução do método (s)
- meanIterationTime: Tempo médio de cada iteração (s)
"""
struct LDCFStats
  code::Symbol
  method::Symbol
  numberOfIterations::Int64
  overheadTime::Float64
  executionTime::Float64
  meanIterationTime::Float64
end

"""
Domínio da simulação
- linMesh: Malha linear utilizada na simulação
- Re: Número de Reynolds utilizado na simulação
- ψ: Corrente
- ω: Vorticidade
- V: Velocidade
- A: Fatoração esparsa utilizada para solução do sistema (LU, LDLt, etc)
- coefs: Coeficientes utilizados no cálculo da velocidade (V = (u, v))
- grids: Offsets utilizados no cálculo da velocidade
"""
mutable struct LDCFDomain
  linMesh::LinearMesh
  ψ::Matrix{Float64}
  ω::Matrix{Float64}
  V::Array{Float64,3}
  A::Union{Nothing,Any}
  coefs::Union{Nothing, Vector{Vector{Float64}}}
  grids::Union{Nothing, Vector{Tuple{Int64,Int64}}}
end

"""
Parâmetros da simulação
- δt: Passo de tempo utilizado na simulação
- tol: Tolerância utilizada na simulação
- maxRes: Resíduo máximo tolerado na simulação
- maxIter: Número máximo de iterações permitidas na simulação
- order: Ordem da aproximação utilizada na simulação
"""
struct LDCFParameters
  Re::Float64
  δt::Float64
  tol::Float64
  maxRes::Float64
  maxIter::Int64
  order::Int
end

"""
Preparação para execução do algoritmo.
- Checagem de tipos
- Construção de espaço linear
- Construção do domínio e agrupamento dos parâmetros de simulação
"""
function prepareSimulation(
  x::Tuple{Tuple{Float64,Float64},Int64},
  y::Tuple{Tuple{Float64,Float64},Int64},
  u₀::Float64)::LDCFDomain
  # Checa se `x` e `y` são crescentes
  xRange, yRange = x[1], y[1]
  @assert xRange[1] < xRange[2] "Intervalo em 'x' não é válido"
  @assert yRange[1] < yRange[2] "Intervalo em 'y' não é válido"

  # Formando coordenadas utilizando espaços lineares
  nx, ny = (x[2], y[2]) .+ 1
  xLinRange = LinRange(xRange[1], xRange[2], nx)
  yLinRange = LinRange(yRange[1], yRange[2], ny)
  δx = xLinRange[2] - xLinRange[1]
  δy = yLinRange[2] - yLinRange[1]
  linMesh = LinearMesh(nx, ny, δx, δy)

  # Alocando espaço para os valores de:
  # ψ - Corrente
  # ω - Vorticidade
  # V - Velocidade
  ψ = zeros(nx, ny)
  ω = zeros(nx, ny)
  V = zeros(nx, ny, 2)
  # Velocidade inicial da tampa móvel
  V[1:nx, ny, 1] .= u₀

  return LDCFDomain(
    linMesh,
    ψ,
    ω,
    V,
    nothing,
    nothing,
    nothing
  )
end

function _updateOuterVorticity!(domain!::LDCFDomain)
  linMesh, ψ, ω, V = domain!.linMesh, domain!.ψ, domain!.ω, domain!.V
  nx, ny = linMesh.nx, linMesh.ny
  δx, δy = linMesh.δx, linMesh.δy

  @inbounds Threads.@threads for i in 2:nx-1
    # Parede inferior
    ω[i, 1] = ((1 / 2) * ψ[i, 3] - 4 * ψ[i, 2]) / δy^2
    # Parede superior
    ω[i, ny] = ((1 / 2) * ψ[i, ny-2] - 4 * ψ[i, ny-1] - 3 * δy) / δy^2
  end

  @inbounds Threads.@threads for j in 2:ny-1
    # Parede esquerda
    ω[1, j] = ((1 / 2) * ψ[3, j] - 4 * ψ[2, j]) / δx^2
    # Parede direita
    ω[nx, j] = ((1 / 2) * ψ[nx-2, j] - 4 * ψ[nx-1, j]) / δx^2
  end
end

"""
Função que atualiza os valores de ω a partir da fórmula:

ω = ω + ∂w/∂t, onde

∂ω/∂t = (∂²ω/∂x² + ∂²ω/∂y²)/Re - u⋅∂ω/∂x - v⋅∂ω/∂y
"""
function updateVorticity!(domain!::LDCFDomain, params::LDCFParameters)
  _updateOuterVorticity!(domain!)

  linMesh, ω₀, V = domain!.linMesh, domain!.ω, domain!.V
  nx, ny = linMesh.nx, linMesh.ny
  δx, δy = linMesh.δx, linMesh.δy
  Re, δt = params.Re, params.δt
  ω = copy(ω₀)

  # Atualizando ω com base nos valores anteriores
  @inbounds Threads.@threads for j in 2:ny-1
    for i in 2:nx-1
      ω[i, j] += δt * (
        (
          (ω₀[i+1, j] - 2 * ω₀[i, j] + ω₀[i-1, j]) / δx^2 +
          (ω₀[i, j+1] - 2 * ω₀[i, j] + ω₀[i, j-1]) / δy^2
        ) / Re -
        V[i, j, 1] * (ω₀[i+1, j] - ω₀[i-1, j]) / (2 * δx) -
        V[i, j, 2] * (ω₀[i, j+1] - ω₀[i, j-1]) / (2 * δy)
      )
    end
  end

  return ω
end
