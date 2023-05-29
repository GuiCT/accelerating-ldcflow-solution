using Plots
using PlotThemes
theme(:default)

struct GhiaSolution
  x::Vector
  y::Vector
  u::Vector
  v::Vector
end

function compareGhia(u, v, Re)
  ghiaSolution = Dict(
    100 => GhiaSolution(
      [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0],
      [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0],
      [1, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0],
      [0, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0]
    ),
    400 => GhiaSolution(
      [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0],
      [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0],
      [1, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0],
      [0, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0]
    ),
    1000 => GhiaSolution(
      [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0],
      [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0],
      [1, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0],
      [0, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0]
    ),
    3200 => GhiaSolution(
      [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0],
      [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0],
      [1, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791, 0.07156, -0.04272, -0.86636, -0.24427, -0.34323, -0.41933, -0.37827, -0.35344, -0.32407, 0],
      [0, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307, -0.37401, -0.31184, 0.00999, 0.28188, 0.29030, 0.37119, 0.42768, 0.41906, 0.40917, 0.39560, 0]
    ),
    10000 => GhiaSolution(
      [1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0],
      [1, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0],
      [1, 0.47221, 0.47783, 0.48070, 0.47804, 0.34635, 0.20673, 0.08344, 0.03111, -0.07540, -0.23186, -0.32709, -0.38000, -0.41657, -0.42537, -0.42735, 0],
      [0, -0.54302, -0.52987, -0.49099, -0.45863, -0.41496, -0.36737, -0.30719, 0.00831, 0.27224, 0.28003, 0.35070, 0.41487, 0.43124, 0.43733, 0.43983, 0]
    )
  )

  nx, ny = size(u) .- 1
  x = LinRange(0, 1, nx + 1)
  y = LinRange(0, 1, ny + 1)

  plotU = plot(
    [
      (ghiaSolution[Re].u, ghiaSolution[Re].y),
      (u[nx÷2+1, :], y)
    ],
    seriestype=[:scatter :path],
    label=["Ghia et. Al." "Solução obtida"],
    title="Valor de u: Ghia et. Al. vs Obtido"
  )
  savefig(plotU, "u.png")

  plotV = plot(
    [
      (ghiaSolution[Re].x, ghiaSolution[Re].v),
      (x, v[:, nx÷2+1])
    ],
    seriestype=[:scatter :path],
    label=["Ghia et. Al." "Solução obtida"],
    title="Valor de v: Ghia et. Al. vs Obtido"
  )
  savefig(plotV, "v.png")
end
