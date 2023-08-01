using LDCFlow
using HDF5

# Por ora, utilizando apenas dois Reynolds: 100.0 e 400.0
# TODO: expandir para todos os Reynolds presentes no Ghia et. Al.
Reynolds = [100.0, 400.0]
n = 64
δt = 0.001

global velocityHistoryFile = h5open("./data_driven/velocityHistory.h5", "w")
global history = Array{Float64, 3}[]

function callback_save_to_history(
  simulationDomain::LDCFDomain,
  simulationParameters::LDCFParameters,
  iterationNumber::Int64,
  V₀::Array{Float64, 3}
  )
  global history
  V = copy(simulationDomain.V)
  push!(history, V)
  # Lidando com resíduos
  residual_u = maximum(abs.(simulationDomain.V[:, :, 1] - V₀[:, :, 1]))
  residual_v = maximum(abs.(simulationDomain.V[:, :, 2] - V₀[:, :, 2]))
  @info "Iteração: $(iterationNumber), Resíduos: $(residual_u), $(residual_v)"
  if (residual_u < simulationParameters.tol &&
      residual_v < simulationParameters.tol)
    return 1 # Sucesso
  elseif (residual_u > simulationParameters.maxRes ||
          residual_v > simulationParameters.maxRes)
    return -1 # Falha
  else
    return 0 # Continuar
  end
end

# Para todos os Reynolds analisados
# Salvar como um tensor de 4 dimensões (tempo, x, y, V)
for reynold ∈ Reynolds
  history = Array{Float64, 3}[]
  LDCF2Order(n,n, reynold, δt; callback=callback_save_to_history)
  velocityHistory = zeros(
    3000,
    size(history[1])[1],
    size(history[1])[2],
    size(history[1])[3]
  )
  # Distribute actual history size to fit in 3000
  # LinRange it and then round
  steps = LinRange(1, length(history), 3000)
  steps = round.(Int, steps)
  for (idx, i) in enumerate(steps)
    velocityHistory[idx, :, :, :] = history[i]
  end
  write(velocityHistoryFile, "Reynolds_$reynold", velocityHistory)
end


close(velocityHistoryFile)