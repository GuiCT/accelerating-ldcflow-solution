using FiniteDifferences;
using LinearAlgebra;
using SparseArrays;

function generate_coeff_matrix(n, δ, points=3, use_descenters=true)
  @assert points % 2 == 1 "N of points must be odd"
  fdm = central_fdm(points, 2)
  coefs = Array(fdm.coefs)
  offsets = Array(fdm.grid)
  a = spdiagm([offsets[i] => fill(coefs[i], n - abs(offsets[i])) for i ∈ 1:points]...)
  if use_descenters
    for i ∈ 1:points÷2
      offset = i - 1
      lower_grid = collect((0:points-1) .- offset)
      upper_grid = reverse(-lower_grid)
      lower_coefs = Array(FiniteDifferenceMethod(lower_grid, 2).coefs)
      upper_coefs = Array(FiniteDifferenceMethod(upper_grid, 2).coefs)
      lower_positions = (1:points) .+ offset
      upper_positions = (n-points+1:n) .- offset
      a[i, lower_positions] = lower_coefs
      a[end-offset, upper_positions] = upper_coefs
    end
  end
  a = a / δ^2
  id = I(n)
  return kron(a, id) + kron(id, a)
end

