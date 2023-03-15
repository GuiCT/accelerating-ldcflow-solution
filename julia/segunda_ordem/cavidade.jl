# Pacote SparseArrays é utilizado para criação de matriz esparsa de Poisson
using SparseArrays;
using LinearAlgebra;

"""
    cavidade(nx::Int, ny::Int, Re::Int, δt = 0.001,
    nt = 10000, xRange = [0, 1], yRange = [0, 1])

Resolve o problema da cavidade com tampa móvel utilizando aproximações de segunda ordem.

# Arguments
- `nx::Int`: número de elementos ao longo do eixo x.
- `ny::Int`: número de elementos ao longo do eixo y.
- `Re::Int`: número de Reynolds.
- `δt`: passo de integração temporal, por padrão 0.001.
- `nt`: quantidade máxima de iterações, por padrão 10000.
- `xRange`: intervalo no eixo x, por padrão de 0 à 1 (unitário).
- `yRange`: intervalo no eixo y, por padrão de 0 à 1 (unitário).

Os argumentos `𝛿t`, `nt`, `xRange` e `yRange` são opcionais.
"""
function cavidade(
  nx::Int, ny::Int, Re::Int, δt=0.001,
  nt::Int=10000, xRange=[0, 1], yRange=[0, 1]
)
  # Espaço linear de x e y a partir da quantidade de elementos e o range especificado
  x = LinRange(xRange[1], xRange[2], nx + 1)
  y = LinRange(yRange[1], yRange[2], ny + 1)

  # δx e δy são calculados a partir da diferença entre dois adjacentes no espaço linear
  δx = x[2] - x[1]
  δy = y[2] - y[1]

  ψ = zeros(nx + 1, ny + 1)      # Corrente
  u = zeros(nx + 1, ny + 1)      # Velocidade
  v = zeros(nx + 1, ny + 1)      # Velocidade
  ω = zeros(nx + 1, ny + 1)      # Vorticidade

  u[1:nx+1, ny+1] .= 1   # Velocidade inicial da tampa

  # Realizando a montagem da matriz de Poisson
  # E armazenando resultado da fatoração LU
  # A fatoração LU é utilizada na resolução do sistema linear esparso
  # Como a matriz é constante durante a execução do código, o resultado
  # Da fatoração LU também é. Sendo a tarefa mais intensiva do processo
  # de resolução do Sistema Linear, isso melhora de forma patente a performance
  # do código em questão.
  A_LU = lu(matrizPoisson(nx, ny, δx, δy))
  # Vetor independente do sistema
  b = zeros((nx + 1) * (ny + 1))

  for iterationNumber in 1:nt
    ω = calculoContorno!(δx, δy, ψ, ω)
    ω = calculoVetorIndependente!(Re, δx, δy, δt, ω, u, v, b)
    ψ = resolucaoSistemaLinear(nx, ny, b, A_LU)
    u₀ = copy(u)
    v₀ = copy(v)
    u, v = atualizandoUeV(δx, δy, ψ, u, v)

    # Calculando resíduos em u e v
    residuoU = maximum(abs.(u - u₀))
    residuoV = maximum(abs.(v - v₀))
    # Printando informações do passo
    println("Passo: ", iterationNumber, "\tResíduo (u): ", residuoU, "\tResíduo (v): ", residuoV)

    # Se um dos erros for maior que 1e+8, aborta.
    # Se o erro de ambos forem menores que 1e-5, logo, convergiu.
    if (residuoU > 1e+8 || residuoV > 1e+8)
      println("Erro maior que 1e+8, abortando...")
      break
    elseif (residuoU < 1e-5 && residuoV < 1e-5)
      println("Convergiu!")
      return u, v
    end
  end
end

function matrizPoisson(nx::Int, ny::Int, δx, δy)
  # Inicializando matriz esparsa
  A = spzeros((nx + 1) * (ny + 1), (nx + 1) * (ny + 1))

  for i in 1:nx+1
    for j in 1:ny+1
      flatIndex = (i - 1) * (ny + 1) + j
      # Contorno -> Preenche diagonal principal com 1 (identidade)
      if (i == 1) || (i == nx + 1) || (j == 1) || (j == ny + 1)
        A[flatIndex, flatIndex] = 1
        # Fora do contorno -> Matriz pentadiagonal
      else
        # Elemento da esquerda
        A[flatIndex, (i-2)*(ny+1)+j] = δx^-2

        # Elemento do centro
        A[flatIndex, (i-1)*(ny+1)+j] = -2 * (δx^-2 + δy^-2)

        # Elemento da direita
        A[flatIndex, (i)*(ny+1)+j] = δx^-2

        # Elemento de baixo
        A[flatIndex, (i-1)*(ny+1)+j-1] = δy^-2

        # Elemento de cima
        A[flatIndex, (i-1)*(ny+1)+j+1] = δy^-2
      end
    end
  end

  return A
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

function calculoVetorIndependente!(Re, δx, δy, δt, ω₀, u, v, b!)
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

  for i in 2:nx-1
    for j in 2:ny-1
      flatIndex = (i - 1) * ny + j
      b![flatIndex] = -ω[i, j]
    end
  end

  return ω
end

function resolucaoSistemaLinear(nx::Int, ny::Int, b, A_LU)
  solucao = A_LU \ b # Resolve o sistema linear 

  # Realizando reshape da solução, atribuindo à variável ψ
  # Transpose é necessário para que a matriz seja row-wise, ao invés de column-wise.
  ψ = transpose(reshape(solucao, (nx + 1, ny + 1)))
  return ψ
end

function atualizandoUeV(δx, δy, ψ, u!, v!)
  nx = size(ψ, 1)
  ny = size(ψ, 2)

  # Atualizando u e v
  # Próximo do contorno, utilizando diferença centrada
  i = 2:nx-1
  j = 2:ny-1

  for i in [2, nx - 1]
    u![i, j] = (ψ[i, j.+1] - ψ[i, j.-1]) / (2 * δy)
    v![i, j] = -(ψ[i+1, j] - ψ[i-1, j]) / (2 * δx)
  end

  for j in [2, ny - 1]
    u![i, j] = (ψ[i, j+1] - ψ[i, j-1]) / (2 * δy)
    v![i, j] = -(ψ[i.+1, j] - ψ[i.-1, j]) / (2 * δx)
  end

  # Para os demais valores, utiliza diferença finita de quarta ordem
  i = 3:nx-2
  j = 3:ny-2

  u![i, j] = (
    2 * ψ[i, j.-2] -
    16 * ψ[i, j.-1] +
    16 * ψ[i, j.+1] -
    2 * ψ[i, j.+2]
  ) / (24 * δy)

  v![i, j] = -(
    2 * ψ[i.-2, j] -
    16 * ψ[i.-1, j] +
    16 * ψ[i.+1, j] -
    2 * ψ[i.+2, j]
  ) / (24 * δx)

  return u!, v!
end
