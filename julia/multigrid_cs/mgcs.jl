struct Grid
  F::Matrix{Float64}
  TF::Matrix{Float64}
  δx::Float64
  δy::Float64
end

function halfGrid(grid::Grid)
  halfSize = (size(grid.F) .+ 1) .÷ 2

  return Grid(
    zeros(halfSize),
    zeros(halfSize),
    2 * grid.δx,
    2 * grid.δy
  )
end

"""
    multigrid_CS(F, TF, δx, δy, tol=1e-5, max_iter=1000)

Aplica o método de Multigrid com restrição e prolongamento em duas malhas
para a solução de uma Equação Diferencial Parcial

# Arguments
- `F`: matriz contendo valores da solução
- `TF`: matriz do termo-fonte
- `δx`: passo de integração no eixo x
- `δy`: passo de integração no eixo y
- `tol`: tolerância permitida para convergência, por padrão 1e-5.
- `max_iter`: número máximo de iterações, por padrão 1000.

Os argumentos `tol` e `max_iter` são opcionais.
"""
function multigrid_CS(
  F::Matrix{Float64}, TF::Matrix{Float64},
  δx::Float64, δy::Float64,
  tol::Float64=1e-5, max_iter::Int64=1000)

  grid_1 = Grid(copy(F), copy(TF), δx, δy)
  residual_1 = zeros(size(grid_1.F))
  grid_2 = halfGrid(grid_1)
  residual_2 = zeros(size(grid_2.F))
  grid_3 = halfGrid(grid_2)
  residual_3 = zeros(size(grid_3.F))
  grid_4 = halfGrid(grid_3)

  for iterationNumber in 1:max_iter
    jacobi_2D!(grid_1)
    calculateResidual!(grid_1, residual_1)

    iter_error = maximum(abs.(residual_1))
    if (iter_error < tol)
      return grid_1.F
    end

    restriction!(grid_2, residual_1)
    jacobi_2D!(grid_2)
    calculateResidual!(grid_2, residual_2)

    restriction!(grid_3, residual_2)
    jacobi_2D!(grid_3)
    calculateResidual!(grid_3, residual_3)

    restriction!(grid_4, residual_3)
    gauss_seidel_2D!(grid_4)

    correction_3 = polynomial_interp_2D(grid_4)
    grid_3.F .+= correction_3
    correction_2 = polynomial_interp_2D(grid_3)
    grid_2.F .+= correction_2
    correction_1 = polynomial_interp_2D(grid_2)
    grid_1.F .+= correction_1
  end
end

"""
Aplica o método de Jacobi em duas dimensões a partir do valor de F e termo-fonte.

Os parâmetros de fator de relaxamento e número máximo de iterações podem ser alterados.
"""
function jacobi_2D!(grid::Grid, relax_factor=0.8, max_iter=3)
  β = grid.δx^2 / grid.δy^2
  nx, ny = size(grid.F)

  for _ in 1:max_iter
    F⁰ = copy(grid.F)
    @inbounds Threads.@threads for i in 2:nx-1
      for j in 2:ny-1
        LDE = (
          F⁰[i+1, j] + F⁰[i-1, j] +
          β * (F⁰[i, j+1] + F⁰[i, j-1]) -
          grid.δx^2 * grid.TF[i, j]
        ) / (2 + 2 * β)
        grid.F[i, j] += relax_factor * (LDE - grid.F[i, j])
      end
    end
  end
end

"""
Aplica o método de Gauss-Seidel em duas dimensões a partir do valor de F e termo-fonte.

Os parâmetros de fator de relaxamento e número máximo de iterações podem ser alterados.
"""
function gauss_seidel_2D!(grid::Grid, relax_factor=1.9, max_iter=200)
  β = grid.δx^2 / grid.δy^2
  nx, ny = size(grid.F)

  for _ in 1:max_iter
    for i in 2:nx-1
      for j in 2:ny-1
        LDE = (
          grid.F[i+1, j] + grid.F[i-1, j] +
          β * (grid.F[i, j+1] + grid.F[i, j-1]) -
          grid.δx^2 * grid.TF[i, j]
        ) / (2 + 2 * β)
        grid.F[i, j] += relax_factor * (LDE - grid.F[i, j])
      end
    end
  end
end

"""
Aplica a operação de restrição em uma determinada malha, reduzindo-a pela metade.

Retorna o termo-fonte utilizado pela nova malha.
"""
function restriction!(grid::Grid, residual::Matrix{Float64})
  nx_half, ny_half = size(grid.TF)

  @inbounds Threads.@threads for i in 2:nx_half-1
    i₂ = 2 * i - 1
    for j in 2:ny_half-1
      j₂ = 2 * j - 1
      grid.TF[i, j] = 0.0625 * (
        4 * residual[i₂, j₂] + 2 * (
          residual[i₂-1, j₂] + residual[i₂+1, j₂] +
          residual[i₂, j₂+1] + residual[i₂, j₂-1]
        ) +
        residual[i₂+1, j₂+1] + residual[i₂+1, j₂-1] +
        residual[i₂-1, j₂-1] + residual[i₂-1, j₂+1]
      )
    end
  end
end

function calculateResidual!(grid::Grid, residual::Matrix{Float64})
  nx, ny = size(residual)

  @inbounds Threads.@threads for j in 2:ny-1
    for i in 2:nx-1
      residual[i, j] = grid.TF[i, j] - (
        ( # ∂²F/∂x²
        grid.F[i+1, j] -
        2 * grid.F[i, j] +
        grid.F[i-1, j]
        ) / grid.δx^2 +
        ( # ∂²F/∂y²
          grid.F[i, j+1] -
          2 * grid.F[i, j] +
          grid.F[i, j-1]
        ) / grid.δy^2
      )
    end
  end
end

"""
    polynomial_interp_2D(F)

Aplica uma interpolação polinomial para realizar a operação de PROLONGAMENTO.

Recebe como parâmetro uma malha mais grossa.

Retorna a correção para a malha mais fina.
"""
function polynomial_interp_2D(grid_coarse::Grid)
  nx, ny = size(grid_coarse.F)

  nx_finer = nx * 2 - 1
  ny_finer = ny * 2 - 1

  correction = zeros(nx_finer, ny_finer)

  correction[1:2:nx_finer, 1:2:ny_finer] = grid_coarse.F[1:(nx_finer+1)÷2, 1:(ny_finer+1)÷2]
  correction[2:2:nx_finer-1, 1:2:ny_finer] = (
    correction[1:2:nx_finer-2, 1:2:ny_finer] +
    correction[3:2:nx_finer, 1:2:ny_finer]
  ) / 2
  correction[1:nx_finer, 2:2:ny_finer-1] = (
    correction[1:nx_finer, 1:2:ny_finer-2] +
    correction[1:nx_finer, 3:2:ny_finer]
  ) / 2

  return correction
end

