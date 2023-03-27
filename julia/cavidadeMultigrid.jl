using SparseArrays;

include("./cavidade_base.jl")
include("./multigrid_cs/mgcs.jl")

"""
    cavidadeMultigrid(nx::Int64, ny::Int64, Re::Int64,
    δt::Float64, nt::Int64 = typemax(Int64), 
    xRange::Tuple{Float64, Float64}=(0.0, 1.0),
    yRange::Tuple{Float64, Float64}=(0.0, 1.0))

Resolve o problema da cavidade com tampa móvel utilizando método Multigrid CS.

# Arguments
- `nx::Int64`: número de elementos ao longo do eixo x.
- `ny::Int64`: número de elementos ao longo do eixo y.
- `Re::Int64`: número de Reynolds.
- `δt::Float64`: passo de integração temporal.
- `nt::Int64`: quantidade máxima de iterações, por padrão o maior valor inteiro de 64 bits.
- `xRange::Tuple{Float64, Float64}`: intervalo no eixo x, por padrão de 0.0 à 1.0 (unitário).
- `yRange::Tuple{Float64, Float64}`: intervalo no eixo y, por padrão de 0.0 à 1.0 (unitário).

Os argumentos `nt`, `xRange` e `yRange` são opcionais.
"""
function cavidadeMultigrid(
  nx::Int64, ny::Int64, Re::Int64,
  δt::Float64, nt::Int64=typemax(Int64),
  xRange::Tuple{Float64, Float64}=(0.0, 1.0),
  yRange::Tuple{Float64, Float64}=(0.0, 1.0)
)::Union{LDCFSolution, Nothing}
  # Inicializando matrizes e domínio do método
  x, y, δx, δy, ψ, u, v, ω = realizaAlocacoes(nx, ny, xRange, yRange)

  for iterationNumber in 1:nt
    ω = calculoContornoω!(δx, δy, ψ, ω)
    ω = atualizandoω(Re, δx, δy, δt, ω, u, v)
    ψ = resolucaoSistemaLinearMultigrid(ψ, ω, δx, δy)
    u₀ = copy(u)
    v₀ = copy(v)
    u, v = atualizandoUVSegundaOrdem!(δx, δy, ψ, u, v)

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

function resolucaoSistemaLinearMultigrid(ψ, ω, δx, δy)
  res = multigrid_CS(ψ, -ω, δx, δy)
  return res
end
