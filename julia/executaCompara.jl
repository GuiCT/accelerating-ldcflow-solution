include("segunda_ordem/cavidade.jl")
include("compareGhia.jl")

using DelimitedFiles

nx = parse(Int, ARGS[1])
ny = parse(Int, ARGS[2])
Re = parse(Int, ARGS[3])
δt = parse(Float64, ARGS[4])
nt = parse(Int, ARGS[5])
solution = cavidade(nx, ny, Re, δt, nt)

if (isnothing(solution))
  exit(1)
end

compareGhia(solution.u, solution.v, Re)

writedlm(
  "u_" * "Re" * string(Re) * "_nx" * string(nx) * "_ny" * string(ny) * "_δt" * string(δt) * ".csv",
  solution.u, ','
)

writedlm(
  "v_" * "Re" * string(Re) * "_nx" * string(nx) * "_ny" * string(ny) * "_δt" * string(δt) * ".csv",
  solution.v, ','
)
