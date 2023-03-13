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
function multigrid_CS(F, TF, δx, δy, tol=1e-5, max_iter=1000)
  for iterationNumber in 1:max_iter
    F = jacobi_2D(F, TF, δx, δy)
    residual1 = zeros(size(F))
    residual1[2:end-1, 2:end-1] = TF[2:end-1, 2:end-1] - (
      ( # ∂²F/∂x²
        F[1:end-2, 2:end-1] -
        2 * F[2:end-1, 2:end-1] +
        F[3:end, 2:end-1]
      ) / δx^2 +
      ( # ∂²F/∂y²
        F[2:end-1, 1:end-2] -
        2 * F[2:end-1, 2:end-1] +
        F[2:end-1, 3:end]
      ) / δy^2
    )

    iter_error = maximum(abs.(residual1))
    if (iter_error < tol)
      return F, iterationNumber
    end

    TF_coarse = restriction(residual1)
    F_coarse = zeros(size(TF_coarse))
    δx_coarse = δx * 2
    δy_coarse = δy * 2

    F_coarse = gauss_seidel_2D(F_coarse, TF_coarse, δx_coarse, δy_coarse)
    correction = polynomial_interp_2D(F_coarse)
    F += correction
  end
end

"""
Aplica o método de Jacobi em duas dimensões a partir do valor de F e termo-fonte.

Os parâmetros de fator de relaxamento e número máximo de iterações podem ser alterados.
"""
function jacobi_2D(F, TF, δx, δy, relax_factor=0.8, max_iter=3)
  β = δx^2 / δy^2
  nx, ny = size(F)

  for _ in 1:max_iter
    F⁰ = copy(F)
    for i in 2:nx-1
      for j in 2:ny-1
        LDE = (-δx^2 * TF[i, j] + F⁰[i+1, j] + F⁰[i-1, j] + β * (F⁰[i, j-1] + F⁰[i, j+1])) / (2 + 2 * β)
        F[i, j] += relax_factor * (LDE - F[i, j])
      end
    end
  end

  return F
end

"""
Aplica o método de Gauss-Seidel em duas dimensões a partir do valor de F e termo-fonte.

Os parâmetros de fator de relaxamento e número máximo de iterações podem ser alterados.
"""
function gauss_seidel_2D(F, TF, δx, δy, relax_factor=1.9, max_iter=100)
  β = δx^2 / δy^2
  nx, ny = size(F)

  for _ in 1:max_iter
    for i in 2:nx-1
      for j in 2:ny-1
        LDE = (-δx^2 * TF[i, j] + F[i+1, j] + F[i-1, j] + β * (F[i, j-1] + F[i, j+1])) / (2 + 2 * β)
        F[i, j] += relax_factor * (LDE - F[i, j])
      end
    end
  end

  return F
end

"""
Aplica a operação de restrição em uma determinada malha, reduzindo-a pela metade.

Retorna o termo-fonte utilizado pela nova malha.
"""
function restriction(residual)
  nx, ny = size(residual)
  nx_half = (nx + 1) ÷ 2
  ny_half = (ny + 1) ÷ 2

  TF_coarse = zeros(nx_half, ny_half)

  for i in 2:nx_half-1
    i₂ = 2 * i - 1
    for j in 2:ny_half-1
      j₂ = 2 * j - 1
      TF_coarse[i, j] = 0.0625 * (
        4 * residual[i₂, j₂] + 2 * (
          residual[i₂-1, j₂] + residual[i₂+1, j₂] +
          residual[i₂, j₂+1] + residual[i₂, j₂-1]
        ) +
        residual[i₂+1, j₂+1] + residual[i₂+1, j₂-1] +
        residual[i₂-1, j₂-1] + residual[i₂-1, j₂+1]
      )
    end
  end

  return TF_coarse
end

"""
    polynomial_interp_2D(F)

Aplica uma interpolação polinomial para realizar a operação de PROLONGAMENTO.

Recebe como parâmetro uma malha mais grossa.

Retorna a correção para a malha mais fina.
"""
function polynomial_interp_2D(F)
  nx, ny = size(F)

  nx_finer = nx * 2 - 1
  ny_finer = ny * 2 - 1

  correction = zeros(nx_finer, ny_finer)

  correction[1:2:nx_finer, 1:2:ny_finer] = F[1:(nx_finer+1)÷2, 1:(ny_finer+1)÷2]
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

