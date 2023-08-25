module LDCFlow

using LinearAlgebra;

include("./LDCFlow_base.jl")
include("./LDCFlow_fdm.jl")
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
  V₀::Array{Float64,3}
)
  residual_u = maximum(abs.(domain.V[:, :, 1] - V₀[:, :, 1]))
  residual_v = maximum(abs.(domain.V[:, :, 2] - V₀[:, :, 2]))
  @info "Iteração $(iterationNumber): Resíduos: $(residual_u), $(residual_v)"
  if (residual_u > parameters.maxRes || residual_v > parameters.maxRes)
    @error "Resíduo máximo atingido, cancelando simulação."
    return :divergenceToHighResidual
  elseif (residual_u < parameters.tol && residual_v < parameters.tol)
    @info "Convergiu dentro da tolerância especificada."
    return :convergenceWithinTolerance
  else
    return :keep
  end
end

function _ldcf_base(
  nx::Int64, ny::Int64,
  Re::Float64, δt::Float64,
  method::Symbol;
  callback::Function=_default_callback,
  xRange::Tuple{Float64,Float64}=(0.0, 1.0),
  yRange::Tuple{Float64,Float64}=(0.0, 1.0),
  u₀::Float64=1.0,
  tol::Float64=1e-5,
  maxRes::Float64=1e+8,
  maxIter::Int64=typemax(Int64),
  order::Int=2
)::Tuple{LDCFSolution,LDCFStats}
  t1_overhead = time()
  if method == :second_order
    order = 2
  elseif method == :fourth_order
    order = 4
  elseif method == :generic_fdm
  else
    throw(ArgumentError("Método não implementado"))
  end

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
    maxIter,
    order
  )

  if method == :second_order
    solveFunction! = systemSolveFDM!
    updateVelocityFunction! = updateVelocity2Order!
    simulationDomain.A = -generateCoefficientMatrix(simulationDomain.linMesh, order)
  elseif method == :fourth_order
    solveFunction! = systemSolve4Order!
    updateVelocityFunction! = updateVelocity4Order!
    simulationDomain.A = -matrix4Order(simulationDomain.linMesh)
  elseif method == :generic_fdm
    solveFunction! = systemSolveFDM!
    updateVelocityFunction! = updateVelocityFDM!
    simulationDomain.A = -generateCoefficientMatrix(simulationDomain.linMesh, order)
    simulationDomain.grids,
    simulationDomain.coefs = generateFDMGridAndCoefficients(order)
  else
    throw(ArgumentError("Método não implementado"))
  end
  ps = MKLPardisoSolver()
  pardisoinit(ps)
  set_iparm!(ps, 12, 1)
  set_phase!(ps, 12)
  set_matrixtype!(ps, 2)
  set_nprocs!(ps, BLAS.get_num_threads())
  pardiso(ps, zeros((nx - 1) * (ny - 1)), tril(simulationDomain.A), zeros((nx - 1) * (ny - 1)))
  set_phase!(ps, 33)
  simulationDomain.ps = ps
  overhead_duration = time() - t1_overhead

  t1_execution = time()
  f_iterationNumber = 0
  f_code = :outOfLoops
  for iterationNumber in 1:simulationParameters.maxIter
    simulationDomain.ω = updateVorticity!(simulationDomain, simulationParameters)
    solveFunction!(simulationDomain)
    V₀ = copy(simulationDomain.V)
    updateVelocityFunction!(simulationDomain)
    code = callback(
      simulationDomain,
      simulationParameters,
      iterationNumber,
      V₀)
    if code != :keep
      f_iterationNumber = iterationNumber
      f_code = code
      break
    end
  end

  if f_code == :outOfLoops
    f_iterationNumber = simulationParameters.maxIter
  end

  execution_duration = time() - t1_execution
  return (
    LDCFSolution(
      simulationDomain.linMesh,
      simulationDomain.V
    ),
    LDCFStats(
      f_code,
      method,
      f_iterationNumber,
      overhead_duration,
      execution_duration,
      execution_duration / f_iterationNumber
    )
  )
end

function LDCF2Order(
  n::Int64, Re::Real, δt::Float64;
  callback::Function=_default_callback,
  range::Tuple{Float64,Float64}=(0.0, 1.0),
  u₀::Float64=1.0,
  tol::Float64=1e-5,
  maxRes::Float64=1e+8,
  maxIter::Int64=typemax(Int64)
)
  return _ldcf_base(
    n, n, Float64(Re), δt, :second_order,
    callback=callback,
    xRange=range,
    yRange=range,
    u₀=u₀,
    tol=tol,
    maxRes=maxRes,
    maxIter=maxIter
  )
end

function LDCF4Order(
  n::Int64, Re::Real, δt::Float64;
  callback::Function=_default_callback,
  range::Tuple{Float64,Float64}=(0.0, 1.0),
  u₀::Float64=1.0,
  tol::Float64=1e-5,
  maxRes::Float64=1e+8,
  maxIter::Int64=typemax(Int64)
)
  return _ldcf_base(
    n, n, Float64(Re), δt, :fourth_order,
    callback=callback,
    xRange=range,
    yRange=range,
    u₀=u₀,
    tol=tol,
    maxRes=maxRes,
    maxIter=maxIter
  )
end

function LDCFNOrder(
  n::Int64, Re::Real, δt::Float64, order::Int;
  callback::Function=_default_callback,
  range::Tuple{Float64,Float64}=(0.0, 1.0),
  u₀::Float64=1.0,
  tol::Float64=1e-5,
  maxRes::Float64=1e+8,
  maxIter::Int64=typemax(Int64)
)
  return _ldcf_base(
    n, n, Float64(Re), δt, :generic_fdm,
    callback=callback,
    xRange=range,
    yRange=range,
    u₀=u₀,
    tol=tol,
    maxRes=maxRes,
    maxIter=maxIter,
    order=order
  )
end

export LDCF2Order, LDCF4Order, LDCFNOrder, LDCFDomain, LDCFParameters, LDCFSolution, LDCFStats
end # module LDCFLow
