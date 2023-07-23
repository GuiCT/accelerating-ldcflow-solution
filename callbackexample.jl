using HDF5
using LDCFlow # Deve ser instalado com Pkg.dev("./LDCFlow")
# Em caso de alterações, deve-se executar Pkg.build("LDCFlow")

# Abrindo arquivo HDF5 para salvar o histórico de velocidades
global velocityHistoryFile = h5open("velocityHistory.h5", "w")
# Criando histórico
global history = Array{Float64, 3}[]

function callback_example(
  simulationDomain::LDCFDomain,
  simulationParameters::LDCFParameters,
  iterationNumber::Int64,
  V₀::Array{Float64, 3}
  )
  # Referenciando variável global
  global history
  # Copiando campo de velocidades para salvar no histórico
  # Caso contrário, ao fim da execução, todos os valores do histórico
  # apontarão para o mesmo endereço de memória
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

# Chamando o método de segunda ordem e passando a função de callback
LDCF2Order(64, 64, 100.0, 0.001; callback=callback_example)
# Transformando Vetor de Tensores 3D em um Tensor 4D.
# O primeiro índice é o número da iteração, os demais são os índices
# do Tensor 3D.
# Isso é necessário para salvar o histórico no arquivo HDF5.
velocityHistory = zeros(
  size(history)[1],
  size(history[1])[1],
  size(history[1])[2],
  size(history[1])[3]
)
for i in 1:size(history)[1]
  velocityHistory[i, :, :, :] = history[i]
end
write(velocityHistoryFile, "velocityHistory", velocityHistory)

close(velocityHistoryFile)