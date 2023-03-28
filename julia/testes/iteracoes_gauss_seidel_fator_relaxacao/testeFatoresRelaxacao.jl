include("./cavidadeMultigrid.jl")

tested_factors = 0.1:0.1:2.0

# nx = ny = 32

# Re = 100
iterationsMG_32_Re100 = [zeros(Int64, 1) for _ in 1:size(tested_factors)[1]]
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(32, 32, 100, 0.001, test_factor, 1000)
    if isnothing(it)
      iterationsMG_32_Re100[i] = typemax(Int64)
    else
      iterationsMG_32_Re100[i] = it
    end
  catch e
    continue
  end
end

# Re = 400
iterationsMG_32_Re400 = [zeros(Int64, 1) for _ in 1:size(tested_factors)[1]]
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(32, 32, 400, 0.001, test_factor, 1000)
    if isnothing(it)
      iterationsMG_32_Re400[i] = typemax(Int64)
    else
      iterationsMG_32_Re400[i] = it
    end
  catch e
    continue
  end
end

# nx = ny = 64

# Re = 100
iterationsMG_64_Re100 = [zeros(Int64, 1) for _ in 1:size(tested_factors)[1]]
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(64, 64, 100, 0.001, test_factor, 1000)
    if isnothing(it)
      iterationsMG_64_Re100[i] = typemax(Int64)
    else
      iterationsMG_64_Re100[i] = it
    end
  catch e
    continue
  end
end

# Re = 400
iterationsMG_64_Re400 = [zeros(Int64, 1) for _ in 1:size(tested_factors)[1]]
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(64, 64, 400, 0.001, test_factor, 1000)
    if isnothing(it)
      iterationsMG_64_Re400[i] = typemax(Int64)
    else
      iterationsMG_64_Re400[i] = it
    end
  catch e
    continue
  end
end

# nx = ny = 128

# Re = 100
iterationsMG_128_Re100 = [zeros(Int64, 1) for _ in 1:size(tested_factors)[1]]
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(128, 128, 100, 0.001, test_factor, 1000)
    if isnothing(it)
      iterationsMG_128_Re100[i] = typemax(Int64)
    else
      iterationsMG_128_Re100[i] = it
    end
  catch e
    continue
  end
end

# Re = 400
iterationsMG_128_Re400 = [zeros(Int64, 1) for _ in 1:size(tested_factors)[1]]
for (i, test_factor) in enumerate(tested_factors)
  try
    _, it = cavidadeMultigrid(128, 128, 400, 0.001, test_factor, 1000)
    if isnothing(it)
      iterationsMG_128_Re400[i] = typemax(Int64)
    else
      iterationsMG_128_Re400[i] = it
    end
  catch e
    continue
  end
end
