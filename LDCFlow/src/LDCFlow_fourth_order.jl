using SparseArrays;

function _generatePatternVector(
  size::Int, default::Number,
  value::Number, each::Int,
  offset::Int=0)::Vector{Float64}

  vector = fill(default, size)
  for i ∈ (each+offset):each:size
    vector[i] = value
  end

  return vector
end

"""
Matriz de coeficientes para o sistema linear esparso de quarta ordem.
"""
function matrix4Order(linMesh::LinearMesh)
  n = linMesh.nx - 1
  return spdiagm(
    -n => _generatePatternVector( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2) * (n - 1) - 1,
      1.0,
      0.0,
      n - 1
    ),
    -(n - 1) => fill(4.0, (n - 2) * (n - 1)), # [4, 4, ..., 4]
    -(n - 2) => _generatePatternVector( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2) * (n - 1) + 1,
      1.0,
      0.0,
      n - 1,
      1
    ),
    -1 => _generatePatternVector( # [4, 4, ..., 0, 4, 4, ...]
      (n - 1)^2 - 1,
      4.0,
      0.0,
      n - 1
    ),
    0 => fill(-20.0, (n - 1)^2), # [-20, -20, ..., -20]
    1 => _generatePatternVector( # [4, 4, ..., 0, 4, 4, ...]
      (n - 1)^2 - 1,
      4.0,
      0.0,
      n - 1
    ),
    n - 2 => _generatePatternVector( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2) * (n - 1) + 1,
      1.0,
      0.0,
      n - 1,
      1
    ),
    n - 1 => fill(4.0, (n - 2) * (n - 1)), # [4, 4, ..., 4]
    n => _generatePatternVector( # [1, 1, ..., 0, 1, 1, ...]
      (n - 2) * (n - 1) - 1,
      1.0,
      0.0,
      n - 1
    )
  )
end

"""
Solução do sistema linear para o problema de Poisson utilizando diferenças
finitas de quarta ordem.
"""
function systemSolve4Order!(domain!::LDCFDomain)
  f, δ, LDLT = -domain!.ω, domain!.linMesh.δx, domain!.A

  totalSize = size(f)[1]
  innerSize = totalSize - 2
  R = zeros(innerSize^2)

  @inbounds Threads.@threads for j ∈ 1:innerSize
    for i ∈ 1:innerSize
      # T[i, j] -> R[(i - 1) * sizeInterno + j] (2D -> 1D)
      R[(i-1)*innerSize+j] = δ^2 * (
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

  domain!.ψ[2:end-1, 2:end-1] = transpose(reshape(x, (innerSize, innerSize)))
end

"""
Atualiza os valores de velocidade a partir da função corrente ψ.
"""
function updateVelocity4Order!(domain!::LDCFDomain)
  linMesh, ψ, V! = domain!.linMesh, domain!.ψ, domain!.V
  nx, ny = linMesh.nx, linMesh.ny
  δx, δy = linMesh.δx, linMesh.δy

  # Quando i ou j são iguais a 2, utiliza diferenças descentradas com
  # grid [-1, 0, 1, 2, 3] e coeficientes [-3, -10, 18, -6, 1] / 12.
  # Quando i ou j estão entre 3 e (nx ou ny)-2, utiliza diferenças centradas
  # com grid [-2, -1, 1, 2] e coeficientes [1, -8, 8, -1] / 12
  # Por fim, quando i ou j são iguais a (nx ou ny)-1, utiliza diferenças descentradas
  # com grid [-3, -2, -1, 0, 1] e coeficientes [-1, 6, -18, 10, 3] / 12 

  # Atualizando u = ∂ψ/∂y
  @inbounds Threads.@threads for i in 2:nx-1
    V![i, 2, 1] = (
      # -3 * ψ[i, 1] == 0
      -10 * ψ[i, 2] +
      18 * ψ[i, 3] -
      6 * ψ[i, 4] +
      ψ[i, 5]
    ) / (12 * δy)

    for j in 3:ny-2
      V![i, j, 1] = (
        ψ[i, j-2] -
        8 * ψ[i, j-1] +
        8 * ψ[i, j+1] -
        ψ[i, j+2]
      ) / (12 * δy)
    end

    V![i, ny-1, 1] = (
      -ψ[i, ny-4] +
      6 * ψ[i, ny-3] -
      18 * ψ[i, ny-2] +
      10 * ψ[i, ny-1]
      # 3 * ψ[i, ny] == 0
    ) / (12 * δy)
  end

  # Atualizando v = - ∂ψ/∂x
  @inbounds Threads.@threads for j in 2:ny-1
    V![2, j, 2] = -(
      # -3 * ψ[1, j] == 0
      -10 * ψ[2, j] +
      18 * ψ[3, j] -
      6 * ψ[4, j] +
      ψ[5, j]
    ) / (12 * δx)

    for i in 3:nx-2
      V![i, j, 2] = -(
        ψ[i-2, j] -
        8 * ψ[i-1, j] +
        8 * ψ[i+1, j] -
        ψ[i+2, j]
      ) / (12 * δx)
    end

    V![nx-1, j, 2] = -(
      -ψ[nx-4, j] +
      6 * ψ[nx-3, j] -
      18 * ψ[nx-2, j] +
      10 * ψ[nx-1, j]
      # 3 * ψ[nx, j] == 0
    ) / (12 * δx)
  end
end
