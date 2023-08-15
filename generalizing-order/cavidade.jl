using SparseArrays;
using LinearAlgebra;

include("./cavidade_base.jl")
include("./form_matrix.jl")

function cavidade(
  n::Int64, Re::Int64,
  δt::Float64, nt::Int64=typemax(Int64),
  num_of_points=3,
  range::Tuple{Float64, Float64}=(0.0, 1.0);
  atol::Float64=1e-6, rtol::Float64=1e-5, limit::Float64=1e+8
)::Union{LDCFSolution, Nothing}
  d, _, δ, _, ψ, u, v, ω = realizaAlocacoes(n, n, range, range)

  factor = factorize(generate_coeff_matrix(n - 1, δ, num_of_points))
  
  for iterationNumber in 1:nt
    ω = calculoContornoω!(δ, δ, ψ, ω)
    ω = atualizandoω(Re, δ, δ, δt, ω, u, v)
    # Atualizando ψ a partir do sistema linear esparso de Poisson
    ψ[2:end-1, 2:end-1] .= resolucaoSistemaLinear(ω, factor)
    u₀ = copy(u)
    v₀ = copy(v)
    u, v = atualizandoUVSegundaOrdem!(δ, δ, ψ, u, v)

    residuoU = maximum(abs.(u - u₀))
    residuoV = maximum(abs.(v - v₀))
    # Printando informações do passo
    println("Passo: ", iterationNumber, "\tResíduo (u): ", residuoU, "\tResíduo (v): ", residuoV)

    if (residuoU > limit || residuoV > limit)
      println("Falhou em atingir convergência.")
      return Nothing
    elseif (residuoU < rtol && residuoV < rtol)
      println("Convergiu dentro da tolerância especificada.")
      return LDCFSolution(
        d, d, u, v, Re, true
      )
    end
  end

  return LDCFSolution(
    d, d, u, v, Re, false
  )
end

function resolucaoSistemaLinear(ω::Matrix{Float64}, factor)  
  ω_inner = -ω[2:end-1, 2:end-1]
  n, _ = size(ω_inner)
  b = Array(reshape(transpose(ω_inner), n * n))
  x = factor \ b
  return transpose(reshape(x, (n, n)))
end
