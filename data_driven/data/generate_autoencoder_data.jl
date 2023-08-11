# ===== Arquivo gerador de dados para o Autoencoder =====
# 1. Declara variáveis globais para guardar os dados
#   - Esses dados persistem entre as iterações do método
# 2. Declara uma função de callback para salvar os dados
#   - Ao fim de cada iteração, é chamada
#   - Salva os dados em um arquivo HDF5
#   - Realiza um snapshot sempre que o resíduo para o snapshot anterior for
#     maior que um limiar
#   - Os dados são salvos como um tensor de 4 dimensões (tempo, x, y, V)
# 3. Chama o método LDCF4Order para gerar os dados
# Ao fim da execução, os dados estarão salvos em um arquivo HDF5
# Serão gerados dados para diferentes valores de Reynolds
# 1000 valores aleatórios entre 20 e 10000
# ============================================================

using LDCFlow
using HDF5

# Gerando dados para multiplos valores de Reynolds
# MUITOS valores entre 20 e 10000, inteiros
tested_reynolds = round.(20 .+ 9980 * rand(1000))
# Removendo valores duplicados, se existirem
tested_reynolds = unique(tested_reynolds)
n = 64
δt = 0.001

global history = Array{Float64,3}[]

function callback_save_to_history(
  simulationDomain::LDCFDomain,
  simulationParameters::LDCFParameters,
  iterationNumber::Int64,
  V₀::Array{Float64,3}
)
  global history
  # Limiar utilizado para o resíduo
  RESIDUAL_THRESHOLD = 10
  # Primeira iteração sempre guarda o valor
  if iterationNumber != 1
    this_it_residual = sum(abs.(simulationDomain.V - history[end]))
    if this_it_residual > RESIDUAL_THRESHOLD
      push!(history, copy(simulationDomain.V))
    end
  else
    push!(history, copy(simulationDomain.V))
  end
  return 0
end

for reynold ∈ tested_reynolds
  global history = Array{Float64,3}[]
  print("\nReynolds: $reynold")
  LDCF4Order(n, reynold, δt; maxIter=10000, callback=callback_save_to_history)
  velocityHistory = zeros(
    length(history),
    size(history[1])[1],
    size(history[1])[2],
    size(history[1])[3]
  )
  for i in eachindex(history)
    velocityHistory[i, :, :, :] = history[i]
  end
  filepath = joinpath(dirname(Base.source_path()), "generated", "autoencoder_data.h5")
  velocityHistoryFile = h5open(filepath, "cw")
  write(velocityHistoryFile, "$reynold", velocityHistory)
  close(velocityHistoryFile)
end
