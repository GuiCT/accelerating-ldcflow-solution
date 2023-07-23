module LDCFlow

include("./LDCFlow_base.jl")
include("./LDCFlow_second_order.jl")
include("./LDCFlow_fourth_order.jl")

"""
Callback padrão. Utiliza resíduo absoluto para determinar se a simulação
deve continuar ou não.
"""
function _default_callback(
  domain::LDCFDomain,
  parameters::LDCFParameters,
  iterationNumber::Int64,
  V₀::Array{Float64, 3}
)
  residual_u = maximum(abs.(domain.V[:, :, 1] - V₀[:, :, 1]))
  residual_v = maximum(abs.(domain.V[:, :, 2] - V₀[:, :, 2]))
  @info "Iteração $(iterationNumber): Resíduos: $(residual_u), $(residual_v)"
  if (residual_u > parameters.maxRes || residual_v > parameters.maxRes)
    @error "Resíduo máximo atingido, cancelando simulação."
    return -1
  elseif (residual_u < parameters.tol && residual_v < parameters.tol)
    @info "Convergiu dentro da tolerância especificada."
    return 1
  else
    return 0
  end
end

function _ldcf_base(
  nx::Int64, ny::Int64,
  Re::Float64, δt::Float64,
  method::Symbol;
  callback::Function=_default_callback,
  xRange::Tuple{Float64, Float64} = (0.0, 1.0),
  yRange::Tuple{Float64, Float64} = (0.0, 1.0),
  u₀::Float64 = 1.0,
  tol::Float64 = 1e-5,
  maxRes::Float64 = 1e+8,
  maxIter::Int64 = typemax(Int64)
)
  simulationDomain = prepareSimulation(
    (xRange, nx),
    (yRange, ny),
    u₀
  )
  simulationParameters = LDCFParameters(
    Re,
    δt,
    tol,
    maxRes,
    maxIter
  )

  if method == :second_order
    solveFunction = systemSolve2Order
    updateVelocityFunction = updateVelocity2Order!
    simulationDomain.A = lu(matrix2Order(simulationDomain.linMesh))
  elseif method == :fourth_order
    solveFunction = systemSolve4Order
    updateVelocityFunction = updateVelocity4Order!
    simulationDomain.A = lu(matrix4Order(simulationDomain.linMesh))
  else
    throw(ArgumentError("Método não implementado, utilize :second_order ou :fourth_order"))
  end

  for iterationNumber in 1:simulationParameters.maxIter
    simulationDomain.ω = updateVorticity(simulationDomain, simulationParameters)
    simulationDomain.ψ = solveFunction(simulationDomain)
    V₀ = copy(simulationDomain.V)
    updateVelocityFunction(simulationDomain)
    code = callback(
      simulationDomain,
      simulationParameters,
      iterationNumber,
      V₀)
    if code != 0
      return LDCFSolution(
        simulationDomain.linMesh,
        simulationDomain.V,
        iterationNumber)
    end
  end

  return LDCFSolution(
    simulationDomain.linMesh,
    simulationDomain.V,
    simulationParameters.maxIter)
end

function LDCF2Order(
  nx::Int64, ny::Int64,
  Re::Float64, δt::Float64;
  callback::Function=_default_callback,
  xRange::Tuple{Float64, Float64} = (0.0, 1.0),
  yRange::Tuple{Float64, Float64} = (0.0, 1.0),
  u₀::Float64 = 1.0,
  tol::Float64 = 1e-5,
  maxRes::Float64 = 1e+8,
  maxIter::Int64 = typemax(Int64)
)
  return _ldcf_base(
    nx, ny, Re, δt, :second_order,
    callback=callback,
    xRange=xRange,
    yRange=yRange,
    u₀=u₀,
    tol=tol,
    maxRes=maxRes,
    maxIter=maxIter
  )
end

function LDCF4Order(
  n::Int64, Re::Float64, δt::Float64;
  callback::Function=_default_callback,
  range::Tuple{Float64, Float64} = (0.0, 1.0),
  u₀::Float64 = 1.0,
  tol::Float64 = 1e-5,
  maxRes::Float64 = 1e+8,
  maxIter::Int64 = typemax(Int64)
)
  return _ldcf_base(
    n, n, Re, δt, :fourth_order,
    callback=callback,
    xRange=range,
    yRange=range,
    u₀=u₀,
    tol=tol,
    maxRes=maxRes,
    maxIter=maxIter
  )
end

export LDCF2Order, LDCF4Order, LDCFDomain, LDCFParameters
end # module LDCFLow
