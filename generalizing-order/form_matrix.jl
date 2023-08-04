using FiniteDifferences;
using LinearAlgebra;
using SparseArrays;

function generate_coeff_matrix(n, δ, points=3)
  # TODO: use non-central differences when dealing with the border
  @assert points % 2 == 1 "N of points must be odd"
  fdm = central_fdm(points, 2)
  coefs = Array(fdm.coefs)
  coefs = coefs ./ (δ^2)
  offsets = Array(fdm.grid)
  a = spdiagm([offsets[i] => fill(coefs[i], n - abs(offsets[i])) for i ∈ 1:points]...)
  id = I(n)
  return kron(a, id) + kron(id, a)
end

