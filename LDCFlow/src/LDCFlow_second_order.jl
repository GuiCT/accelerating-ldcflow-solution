using SparseArrays
using LinearAlgebra

"""
Matriz do sistema linear para o problema de Poisson utilizada na resolução
utilizando diferenças finitas de segunda ordem.
"""
function matrix2Order(linMesh::LinearMesh)
  nx, ny = (linMesh.nx, linMesh.ny) .- 1
  δx, δy = linMesh.δx, linMesh.δy

  numElements = 5 * (nx * ny + 1) - 3 * (nx + ny)
  # Vetores contendo posições de linha, coluna e o elemento em si
  linesIdx = zeros(Int64, numElements)
  columnsIdx = zeros(Int64, numElements)
  elementsValues = zeros(Float64, numElements)
  idx = 1

  for i in 1:nx+1
    for j in 1:ny+1
      flatIndex = (i - 1) * (ny + 1) + j
      # Contorno -> Preenche diagonal principal com 1 (identidade)
      if (i == 1) || (i == nx + 1) || (j == 1) || (j == ny + 1)
        linesIdx[idx] = flatIndex
        columnsIdx[idx] = flatIndex
        elementsValues[idx] = 1
        idx += 1
        # Fora do contorno -> Matriz pentadiagonal
      else
        # Elemento da esquerda
        linesIdx[idx] = flatIndex
        columnsIdx[idx] = (i - 2) * (ny + 1) + j
        elementsValues[idx] = δx^-2
        idx += 1

        # Elemento do centro
        linesIdx[idx] = flatIndex
        columnsIdx[idx] = (i - 1) * (ny + 1) + j
        elementsValues[idx] = -2 * (δx^-2 + δy^-2)
        idx += 1

        # Elemento da direita
        linesIdx[idx] = flatIndex
        columnsIdx[idx] = (i) * (ny + 1) + j
        elementsValues[idx] = δx^-2
        idx += 1

        # Elemento de baixo
        linesIdx[idx] = flatIndex
        columnsIdx[idx] = (i - 1) * (ny + 1) + j - 1
        elementsValues[idx] = δy^-2
        idx += 1

        # Elemento de cima
        linesIdx[idx] = flatIndex
        columnsIdx[idx] = (i - 1) * (ny + 1) + j + 1
        elementsValues[idx] = δy^-2
        idx += 1
      end
    end
  end

  return sparse(linesIdx, columnsIdx, elementsValues)
end

"""
Solução do sistema linear para o problema de Poisson utilizando diferenças
finitas de segunda ordem.
"""
function systemSolve2Order!(domain!::LDCFDomain)
  linMesh, ω, LU = domain!.linMesh, domain!.ω, domain!.A
  nx, ny = linMesh.nx, linMesh.ny
  b = zeros(nx * ny)
  # Formando vetor independente b a partir de ω
  # No contorno -> 0.0
  # Internamente -> -ω
  @inbounds Threads.@threads for j ∈ 2:ny-1
    for i ∈ 2:nx-1
      b[(i-1)*ny+j] = -ω[i, j]
    end
  end
  x = LU \ b
  domain!.ψ = transpose(reshape(x, (ny, nx)))
end

"""
Atualiza os valores de velocidade a partir da função corrente ψ.
"""
function updateVelocity2Order!(domain!::LDCFDomain)
  linMesh, ψ, V! = domain!.linMesh, domain!.ψ, domain!.V
  nx, ny = linMesh.nx, linMesh.ny
  δx, δy = linMesh.δx, linMesh.δy

  @inbounds Threads.@threads for i in 2:nx-1
    for j in 2:ny-1
      V![i, j, 1] = (ψ[i, j+1] - ψ[i, j-1]) / (2 * δy)
      V![i, j, 2] = -(ψ[i+1, j] - ψ[i-1, j]) / (2 * δx)
    end
  end
end
