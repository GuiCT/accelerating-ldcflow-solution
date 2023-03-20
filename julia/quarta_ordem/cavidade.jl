# Pacote SparseArrays é utilizado para criação de matriz esparsa de Poisson
using SparseArrays;
using LinearAlgebra;
using StatsBase;

struct LDCFSolution
  x::LinRange
  y::LinRange
  u::Matrix
  v::Matrix
  converge::Bool
end

"""
    cavidade(nx::Int, ny::Int, Re::Int, δt = 0.001,
    nt = 10000, xRange = [0, 1], yRange = [0, 1])

Resolve o problema da cavidade com tampa móvel utilizando aproximações de segunda ordem.

# Arguments
- `n::Int`: número de elementos ao longo dos eixo x e y.
- `Re::Int`: número de Reynolds.
- `δt`: passo de integração temporal, por padrão 0.001.
- `nt`: quantidade máxima de iterações, por padrão 10000.
- `xRange`: intervalo no eixo x, por padrão de 0 à 1 (unitário).
- `yRange`: intervalo no eixo y, por padrão de 0 à 1 (unitário).

Os argumentos `δt`, `nt`, `xRange` e `yRange` são opcionais.
"""
function cavidade(
  n::Int, Re::Int, δt=0.001,
  nt::Int=10000, xRange=[0, 1], yRange=[0, 1]
)::Union{LDCFSolution, Nothing}
  # Espaço linear de x e y a partir da quantidade de elementos e o range especificado
  x = LinRange(xRange[1], xRange[2], n + 1)
  y = LinRange(yRange[1], yRange[2], n + 1)

  # δx e δy são calculados a partir da diferença entre dois adjacentes no espaço linear
  δx = x[2] - x[1]
  δy = y[2] - y[1]

  ψ = zeros(n + 1, n + 1)      # Corrente
  u = zeros(n + 1, n + 1)      # Velocidade
  v = zeros(n + 1, n + 1)      # Velocidade
  ω = zeros(n + 1, n + 1)      # Vorticidade

  u[1:n+1, n+1] .= 1   # Velocidade inicial da tampa

  # Realizando a montagem da matriz de Poisson

  A = ldlt(matrizPoisson(n))

  for iterationNumber in 1:nt
    ω = calculoContorno!(δx, δy, ψ, ω)
    ω = atualizaω(Re, δx, δy, δt, ω, u, v)
    ψ = resolucaoSistemaLinear(n, δx, -ω, A)
    u₀ = copy(u)
    v₀ = copy(v)
    u, v = atualizandoUeV_4a_ordem(δx, δy, ψ, u, v)

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

function patternVectorGenerate(size::Int, default::Number, value::Number, each::Int, offset::Int = 0)
  vector = fill(default, size)
  for i ∈ (each + offset):each:size
    vector[i] = value
  end

  return vector
end

function matrizPoisson(n::Int)
  # Inicializando matriz esparsa
  return spdiagm(
    - n => patternVectorGenerate( # [1, 1, ..., 0, 1, 1, ...]
        (n - 2)  * (n - 1) - 1,
        1.0,
        0.0,
        n - 1
      ),
    - (n - 1) => fill(4.0, (n - 2) * (n - 1)), # [4, 4, ..., 4]
    - (n - 2) => patternVectorGenerate( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2)  * (n - 1) + 1,
      1.0,
      0.0,
      n - 1,
      1
    ),
    -1 => patternVectorGenerate( # [4, 4, ..., 0, 4, 4, ...]
      (n - 1)^2 - 1,
      4.0,
      0.0,
      n - 1
    ),
    0 => fill(-20.0, (n - 1)^2), # [-20, -20, ..., -20]
    1 => patternVectorGenerate( # [4, 4, ..., 0, 4, 4, ...]
      (n - 1)^2 - 1,
      4.0,
      0.0,
      n - 1
    ),
    n - 2 => patternVectorGenerate( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2)  * (n - 1) + 1,
      1.0,
      0.0,
      n - 1,
      1
    ),
    n - 1 => fill(4.0, (n - 2) * (n - 1)), # [4, 4, ..., 4]
    n => patternVectorGenerate( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2)  * (n - 1) - 1,
      1.0,
      0.0,
      n - 1
    )
  )
end

function calculoContorno!(δx, δy, ψ, ω!)
  nx, ny = size(ω!)

  # Parede superior
  i = 1:nx
  j = ny
  ω![i, j] = (-3 * δy .+ (7 / 2) * ψ[i, j] - 4 * ψ[i, j-1] + (1 / 2) * ψ[i, j-2]) / (δy^(2))

  # Parede inferior
  j = 1
  ω![i, j] = ((7 / 2) * ψ[i, j] - 4 * ψ[i, j+1] + (1 / 2) * ψ[i, j+2]) / (δy^(2))

  # Parede esquerda
  i = 1
  j = 1:ny
  ω![i, j] = ((7 / 2) * ψ[i, j] - 4 * ψ[i+1, j] + (1 / 2) * ψ[i+2, j]) / (δx^(2))

  # Parede direita
  i = nx
  ω![i, j] = ((7 / 2) * ψ[i, j] - 4 * ψ[i-1, j] + (1 / 2) * ψ[i-2, j]) / (δx^(2))

  return ω!
end

function atualizaω(Re, δx, δy, δt, ω₀, u, v) 
  nx, ny = size(ω₀)

  # Atualizando ω com base nos valores anteriores
  i = 2:nx-1
  j = 2:ny-1

  ω = copy(ω₀)
  # ω = ω + ∂ω/∂t
  # ∂ω/∂t = (1/Re)*∇²ω - u * ∂ω/∂x - v * ∂ω/∂y
  # = (∂²ω/∂x² + ∂²ω/∂y²)/Re - u * ∂ω/∂x - v * ∂ω/∂y
  ω[i, j] += δt * (
    # (∂²ω/∂x² + ∂²ω/∂y²)/Re
    (
      # ∂²ω/∂x²
      (ω₀[i.+1, j] - 2 * ω₀[i, j] + ω₀[i.-1, j]) / δx^2 +
      # ∂²ω/∂y²
      (ω₀[i, j.+1] - 2 * ω₀[i, j] + ω₀[i, j.-1]) / δy^2
    ) / Re -
    # u * ∂ω/∂x
    u[i, j] .* (ω₀[i.+1, j] - ω₀[i.-1, j]) / (2 * δx) -
    # v * ∂ω/∂y
    v[i, j] .* (ω₀[i, j.+1] - ω₀[i, j.-1]) / (2 * δy)
  )

  return ω
end

function resolucaoSistemaLinear(n::Int, δ, f, A)
  T = zeros(n - 1, n - 1)

  for j ∈ 1:n-1
    for i ∈ 1:n-1
      T[i, j] = δ^2 * (
        f[i+2, j+1] +
        f[i, j+1] +
        8 * f[i+1, j+1] +
        f[i+1, j+2]+
        f[i+1, j]
      ) / 2
    end
  end

  R = Vector(reshape(transpose(T), (n - 1)^2))

  solucao = A \ R # Resolve o sistema linear 

  # Realizando reshape da solução, atribuindo à variável ψ
  # Transpose é necessário para que a matriz seja row-wise, ao invés de column-wise.
  ψ = zeros(n + 1, n + 1)
  ψ[2:n, 2:n] = transpose(reshape(solucao, (n - 1, n - 1)))
  return ψ
end

function atualizandoUeV_2a_ordem(δx, δy, ψ, u!, v!)
  nx = size(ψ, 1)
  ny = size(ψ, 2)

  @inbounds Threads.@threads for i in 2:nx-1
    for j in 2:ny-1
      u![i, j] = (ψ[i, j+1] - ψ[i, j-1]) / (2 * δy)
      v![i, j] = -(ψ[i+1, j] - ψ[i-1, j]) / (2 * δx)
    end
  end

  return u!, v!
end

function atualizandoUeV_4a_ordem(δx, δy, ψ, u!, v!)
  nx = size(ψ, 1)
  ny = size(ψ, 2)

  # Quando i ou j são iguais a 2, utiliza diferenças descentradas com
  # grid [-1, 0, 1, 2, 3] e coeficientes [-3, -10, 18, -6, 1] / 12.
  # Quando i ou j estão entre 3 e (nx ou ny)-2, utiliza diferenças centradas
  # com grid [-2, -1, 1, 2] e coeficientes [1, -8, 8, -1] / 12
  # Por fim, quando i ou j são iguais a (nx ou ny)-1, utiliza diferenças descentradas
  # com grid [-3, -2, -1, 0, 1] e coeficientes [-1, 6, -18, 10, 3] / 12 

  # Atualizando u = ∂ψ/∂y
  i = 2:nx-1
  @inbounds Threads.@threads for j in 2:ny-1
    if j == 2
      u![i, j] = (
        -3 * ψ[i, j-1] -
        10 * ψ[i, j] +
        18 * ψ[i, j+1] -
        6 * ψ[i, j+2] +
        ψ[i, j+3]
      ) / (12 * δy)
    elseif j < ny - 1
      u![i, j] = (
        ψ[i, j-2] -
        8 * ψ[i, j-1] +
        8 * ψ[i, j+1] -
        ψ[i, j+2]
      ) / (12 * δy)
    else
      u![i, j] = (
        -ψ[i, j-3] +
        6 * ψ[i, j-2] -
        18 * ψ[i, j-1] +
        10 * ψ[i, j] +
        3 * ψ[i, j+1]
      ) / (12 * δy)
    end
  end

  # Atualizando v = - ∂ψ/∂x
  j = 2:ny-1
  @inbounds Threads.@threads for i in 2:nx-1
    if i == 2
      v![i, j] = -(
        -3 * ψ[i-1, j] -
        10 * ψ[i, j] +
        18 * ψ[i+1, j] -
        6 * ψ[i+2, j] +
        ψ[i+3, j]
      ) / (12 * δx)
    elseif i < nx - 1
      v![i, j] = -(
        ψ[i-2, j] -
        8 * ψ[i-1, j] +
        8 * ψ[i+1, j] -
        ψ[i+2, j]
      ) / (12 * δx)
    else
      v![i, j] = -(
        -ψ[i-3, j] +
        6 * ψ[i-2, j] -
        18 * ψ[i-1, j] +
        10 * ψ[i, j] +
        3 * ψ[i+1, j]
      ) / (12 * δx)
    end
  end

  return u!, v!
end
