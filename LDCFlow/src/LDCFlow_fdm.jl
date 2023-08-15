using FiniteDifferences;
using SparseArrays;
using LinearAlgebra;
using StaticArrays;

"""
Coeficientes utilizados para formar matrizes para o método de Diferenças Finitas
"""
const COEFS_UNTIL_14 = Dict(
  2 => [1.0, -2.0, 1.0],
  4 => [-0.08333333333333333, 1.3333333333333333, -2.5, 1.3333333333333333, -0.08333333333333333],
  6 => [0.011111111111111112, -0.15, 1.5, -2.7222222222222223, 1.5, -0.15, 0.011111111111111112],
  8 => [-0.0017857142857142857, 0.025396825396825397, -0.2, 1.6, -2.8472222222222223, 1.6, -0.2, 0.025396825396825397, -0.0017857142857142857],
  10 => [0.00031746031746031746, -0.00496031746031746, 0.03968253968253968, -0.23809523809523808, 1.6666666666666667, -2.9272222222222224, 1.6666666666666667, -0.23809523809523808, 0.03968253968253968, -0.00496031746031746, 0.00031746031746031746],
  12 => [-6.012506012506013e-5, 0.001038961038961039, -0.008928571428571428, 0.05291005291005291, -0.26785714285714285, 1.7142857142857142, -2.9827777777777778, 1.7142857142857142, -0.26785714285714285, 0.05291005291005291, -0.008928571428571428, 0.001038961038961039, -6.012506012506013e-5],
  14 => [1.1892869035726179e-5, -0.00022662522662522663, 0.0021212121212121214, -0.013257575757575758, 0.06481481481481481, -0.2916666666666667, 1.75, -3.02359410430839, 1.75, -0.2916666666666667, 0.06481481481481481, -0.013257575757575758, 0.0021212121212121214, -0.00022662522662522663, 1.1892869035726179e-5],
)

"""
Função que gera a matriz para resolução da Equação de Poisson
utilizando determinada ordem.
"""
function generateCoefficientMatrix(linMesh::LinearMesh, order::Int8)
  n = linMesh.nx - 2
  δ = linMesh.δx
  @assert order % 2 == 0 "Ordem deve ser par"
  coefs = if order <= 14
    COEFS_UNTIL_14[order]
  else
    fdm = central_fdm(order + 1, 2)
    Array(fdm.coefs)
  end
  offsets = -(order ÷ 2):(order÷2)
  a = spdiagm([offsets[i] => fill(coefs[i], n - abs(offsets[i])) for i ∈ eachindex(offsets)]...)
  a = a / δ^2
  id = I(n)
  return kron(a, id) + kron(id, a)
end

"""
Solução do sistema linear para o problema de Poisson utilizando diferenças
finitas.
"""
function systemSolveFDM!(domain!::LDCFDomain)
  linMesh, ω, LDLT = domain!.linMesh, domain!.ω, domain!.A
  nx, ny = linMesh.nx, linMesh.ny
  b = copy(reshape(transpose(-ω[2:nx-1, 2:ny-1]), (nx - 2) * (ny - 2)))
  x = LDLT \ b
  domain!.ψ[2:nx-1, 2:ny-1] .= transpose(reshape(x, (ny - 2, nx - 2)))
end

function generateFDMGridAndCoefficients(order::Int8)
  grids = Vector{Tuple{Int64,Int64}}(undef, order - 1)
  for i in 1:order-1
    grids[i] = (-i, order - i)
  end

  coefs = Vector{SVector{order + 1,Float64}}(undef, order - 1)
  for i in eachindex(grids)
    fdm = FiniteDifferenceMethod(grids[i][1]:grids[i][2], 1)
    coefs[i] = Array(fdm.coefs)
  end

  return grids, coefs
end

function updateVelocityFDM!(domain!::LDCFDomain)
  linMesh, ψ, V! = domain!.linMesh, domain!.ψ, domain!.V
  coefs, grids = domain!.coefs, domain!.grids
  nx, ny = linMesh.nx, linMesh.ny
  δx, δy = linMesh.δx, linMesh.δy
  num_grids = length(grids)
  middle_grid = num_grids ÷ 2

  # Atualizando u = ∂ψ/∂y
  @inbounds Threads.@threads for i in 2:nx-1
    for k in 1:middle_grid
      V![i, 1+k, 1] = dot(
        ψ[i, (1+k)+grids[k][1]:(1+k)+grids[k][2]],
        coefs[k]
      ) / δy
    end

    for j in (2+middle_grid):(ny-1-middle_grid)
      V![i, j, 1] = dot(
        ψ[i, j+grids[middle_grid+1][1]:j+grids[middle_grid+1][2]],
        coefs[middle_grid+1]
      ) / δy
    end

    for k in (middle_grid+2):num_grids
      V![i, ny-1-num_grids+k, 1] = dot(
        ψ[i, (ny-1-num_grids+k)+grids[k][1]:(ny-1-num_grids+k)+grids[k][2]],
        coefs[k]
      ) / δy
    end
  end

  # Atualizando v = - ∂ψ/∂x
  @inbounds Threads.@threads for j in 2:ny-1
    for k in 1:middle_grid
      V![1+k, j, 2] = -dot(
        ψ[(1+k)+grids[k][1]:(1+k)+grids[k][2], j],
        coefs[k]
      ) / δx
    end

    for i in (2+middle_grid):(ny-1-middle_grid)
      V![i, j, 2] = -dot(
        ψ[i+grids[middle_grid+1][1]:i+grids[middle_grid+1][2], j],
        coefs[middle_grid+1]
      ) / δx
    end

    for k in (middle_grid+2):num_grids
      V![nx-1-num_grids+k, j, 2] = -dot(
        ψ[(nx-1-num_grids+k)+grids[k][1]:(nx-1-num_grids+k)+grids[k][2], j],
        coefs[k]
      ) / δx
    end
  end
end
