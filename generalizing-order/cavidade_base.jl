# Arquivo contendo funções compartilhadas pelos métodos de resolução do
# Problema da Cavidade com Tampa Móvel

"""
Estrutura para armazenar resultados obtidos
"""
struct LDCFSolution
  x::LinRange
  y::LinRange
  u::Matrix
  v::Matrix
  Reynolds::Int64
  convergence::Bool
end

"""
- Inicializa vetores `x` e `y` contendo espaços lineares,
- Calcula `δx` e `δy`
- Aloca memória para as matrizes `ψ`, `u`, `v` e `ω`
- Define velocidade inicial da tampa na matriz `u`

Retorna `x, y, δx, δy, ψ, u, v, ω`
"""
function realizaAlocacoes(
  nx::Int64, ny::Int64,
  xRange::Tuple{Float64, Float64}, yRange::Tuple{Float64, Float64})
  # Garantindo que xRange e yRange são intervalos válidos.
  # Isto é, o máximo é maior que o mínimo.
  @assert xRange[2] > xRange[1] "x_range não é um intervalo válido"
  @assert yRange[2] > yRange[1] "y_range não é um intervalo válido"
  # Garantindo que nx e ny são estritamente positivos
  @assert nx > 0 "nx não é estritamente positivo"
  @assert ny > 0 "ny não é estritamente positivo"

  # Espaço linear de x e y a partir da quantidade de elementos e o range especificado
  x = LinRange(xRange[1], xRange[2], nx + 1)
  y = LinRange(yRange[1], yRange[2], ny + 1)

  # δx e δy são calculados a partir da diferença entre dois adjacentes no espaço linear
  δx = x[2] - x[1]
  δy = y[2] - y[1]

  ψ = zeros(nx + 1, ny + 1) # Corrente
  u = zeros(nx + 1, ny + 1) # Velocidade (primeira componente)
  v = zeros(nx + 1, ny + 1) # Velocidade (segunda componente)
  ω = zeros(nx + 1, ny + 1) # Vorticidade

  u[1:nx+1, ny+1] .= 1   # Velocidade inicial da tampa

  return x, y, δx, δy, ψ, u, v, ω
end

"""
Função que atualiza o valor de ω nos contornos.
"""
function calculoContornoω!(δx, δy, ψ, ω!)
  nx, ny = size(ω!)

  @inbounds Threads.@threads for i in 2:nx-1
    # Parede inferior
    ω![i, 1] = ((1 / 2) * ψ[i, 3] - 4 * ψ[i, 2]) / δy^2
    # Parede superior
    ω![i, ny] = ((1 / 2) * ψ[i, ny-2] - 4 * ψ[i, ny-1] - 3 * δy) / δy^2
  end

  @inbounds Threads.@threads for j in 2:ny-1
    # Parede esquerda
    ω![1, j] = ((1 / 2) * ψ[3, j] - 4 * ψ[2, j]) / δx^2
    # Parede direita
    ω![nx, j] = ((1 / 2) * ψ[nx-2, j] - 4 * ψ[nx-1, j]) / δx^2
  end

  return ω!
end

"""
Função que atualiza os valores de ω a partir da fórmula:

ω = ω + ∂w/∂t, onde

∂ω/∂t = (∂²ω/∂x² + ∂²ω/∂y²)/Re - u⋅∂ω/∂x - v⋅∂ω/∂y
"""
function atualizandoω(Re, δx, δy, δt, ω₀, u, v)
  nx, ny = size(ω₀)

  ω = copy(ω₀)

  # Atualizando ω com base nos valores anteriores
  @inbounds Threads.@threads for j in 2:ny-1
    for i in 2:nx-1
      ω[i, j] += δt * (
        (
          (ω₀[i+1, j] - 2 * ω₀[i, j] + ω₀[i-1, j]) / δx^2 +
          (ω₀[i, j+1] - 2 * ω₀[i, j] + ω₀[i, j-1]) / δy^2
        ) / Re -
        u[i, j] * (ω₀[i+1, j] - ω₀[i-1, j]) / (2 * δx) -
        v[i, j] * (ω₀[i, j+1] - ω₀[i, j-1]) / (2 * δy)
      )
    end
  end

  return ω
end

"""
Função que calcula `u` e `v` a partir de derivadas parciais
em torno do valor de uma função corrente ψ.

Utiliza diferenças finitas centradas de **segunda ordem**.
"""
function atualizandoUVSegundaOrdem!(δx, δy, ψ, u!, v!)
  nx, ny = size(ψ)

  @inbounds Threads.@threads for i in 2:nx-1
    for j in 2:ny-1
      u![i, j] = (ψ[i, j+1] - ψ[i, j-1]) / (2 * δy)
      v![i, j] = -(ψ[i+1, j] - ψ[i-1, j]) / (2 * δx)
    end
  end

  return u!, v!
end

"""
Função que calcula `u` e `v` a partir de derivadas parciais
em torno do valor de uma função corrente ψ.

Utiliza diferenças finitas centradas de **quarta ordem**.
"""
function atualizandoUVQuartaOrdem!(δx, δy, ψ, u!, v!)
  nx, ny = size(ψ)

  # Quando i ou j são iguais a 2, utiliza diferenças descentradas com
  # grid [-1, 0, 1, 2, 3] e coeficientes [-3, -10, 18, -6, 1] / 12.
  # Quando i ou j estão entre 3 e (nx ou ny)-2, utiliza diferenças centradas
  # com grid [-2, -1, 1, 2] e coeficientes [1, -8, 8, -1] / 12
  # Por fim, quando i ou j são iguais a (nx ou ny)-1, utiliza diferenças descentradas
  # com grid [-3, -2, -1, 0, 1] e coeficientes [-1, 6, -18, 10, 3] / 12 

  # Atualizando u = ∂ψ/∂y
  @inbounds Threads.@threads for i in 2:nx-1
    u![i, 2] = (
      # -3 * ψ[i, 1] == 0
      - 10 * ψ[i, 2] +
      18 * ψ[i, 3] -
      6 * ψ[i, 4] +
      ψ[i, 5]
    ) / (12 * δy)

    for j in 3:ny-2
      u![i, j] = (
        ψ[i, j-2] -
        8 * ψ[i, j-1] +
        8 * ψ[i, j+1] -
        ψ[i, j+2]
      ) / (12 * δy)
    end

    u![i, ny-1] = (
        -ψ[i, ny-4] +
        6 * ψ[i, ny-3] -
        18 * ψ[i, ny-2] +
        10 * ψ[i, ny-1]
        # 3 * ψ[i, ny] == 0
      ) / (12 * δy)
  end

  # Atualizando v = - ∂ψ/∂x
  @inbounds Threads.@threads for j in 2:ny-1
    v![2, j] = -(
      # -3 * ψ[1, j] == 0
      - 10 * ψ[2, j] +
      18 * ψ[3, j] -
      6 * ψ[4, j] +
      ψ[5, j]
    ) / (12 * δx)

    for i in 3:nx-2
      v![i, j] = -(
        ψ[i-2, j] -
        8 * ψ[i-1, j] +
        8 * ψ[i+1, j] -
        ψ[i+2, j]
      ) / (12 * δx)
    end

    v![nx-1, j] = -(
      -ψ[nx-4, j] +
      6 * ψ[nx-3, j] -
      18 * ψ[nx-2, j] +
      10 * ψ[nx-1, j]
      # 3 * ψ[nx, j] == 0
    ) / (12 * δx)
  end

  return u!, v!
end
