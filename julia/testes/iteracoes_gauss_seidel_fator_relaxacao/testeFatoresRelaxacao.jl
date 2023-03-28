include("./cavidadeMultigrid.jl")

tested_factors = 0.1:0.1:2.0
iterationsGS = zeros(Int64, size(tested_factors))
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(64, 64, 100, 0.001, test_factor)
    if isnothing(it)
      iterationsGS[i] = typemax(Int64)
    else
      iterationsGS[i] = it
    end
  catch e
    iterationsGS[i] = typemax(Int64)
    continue
  end
end

iterationsGS128 = zeros(Int64, size(tested_factors))
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(128, 128, 100, 0.001, test_factor)
    if isnothing(it)
      iterationsGS128[i] = typemax(Int64)
    else
      iterationsGS128[i] = it
    end
  catch e
    iterationsGS128[i] = typemax(Int64)
    continue
  end
end

 
iterationsGS64Re400 = zeros(Int64, size(tested_factors))
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(64, 64, 400, 0.001, test_factor)
    if isnothing(it)
      iterationsGS64Re400[i] = typemax(Int64)
    else
      iterationsGS64Re400[i] = it
    end
  catch e
    iterationsGS64Re400[i] = typemax(Int64)
    continue
  end
end

iterationsGS128Re400 = zeros(Int64, size(tested_factors))
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(128, 128, 400, 0.001, test_factor)
    if isnothing(it)
      iterationsGS128Re400[i] = typemax(Int64)
    else
      iterationsGS128Re400[i] = it
    end
  catch e
    iterationsGS128Re400[i] = typemax(Int64)
    continue
  end
end