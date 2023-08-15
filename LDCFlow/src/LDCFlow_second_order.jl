using SparseArrays;
using LinearAlgebra;

"""
Atualiza os valores de velocidade a partir da função corrente ψ.
Utiliza diferenças finitas centradas de segunda ordem.
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
