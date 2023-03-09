# Pacote SparseArrays é utilizado para criação de matriz esparsa de Poisson
using SparseArrays;

"""
    cavidade(nx::Int, ny::Int, Re::Int, δt = 0.001,
    nt = 10000, xRange = [0, 1], yRange = [0, 1])

Resolve o problema da cavidade com tampa móvel utilizando aproximações de segunda ordem.

# Arguments
- `nx::Int`: número de elementos ao longo do eixo x.
- `ny::Int`: número de elementos ao longo do eixo y.
- `Re::Int`: número de Reynolds.
- `δt`: passo de integração temporal, por padrão 0.001.
- `nt`: quantidade máxima de iterações, por padrão 10000.
- `xRange`: intervalo no eixo x, por padrão de 0 à 1 (unitário).
- `yRange`: intervalo no eixo y, por padrão de 0 à 1 (unitário).

Os argumentos `𝛿t`, `nt`, `xRange` e `yRange` são opcionais.
"""
function cavidade(
    nx::Int, ny::Int, Re::Int, δt=0.001,
    nt::Int=10000, xRange=[0, 1], yRange=[0, 1]
)
    # Espaço linear de x e y a partir da quantidade de elementos e o range especificado
    x = LinRange(xRange[1], xRange[2], nx + 1)
    y = LinRange(yRange[1], yRange[2], ny + 1)

    # δx e δy são calculados a partir da diferença entre dois adjacentes no espaço linear
    δx = x[2] - x[1]
    δy = y[2] - y[1]

    ψ = zeros(nx + 1, ny + 1)      # Corrente
    u = zeros(nx + 1, ny + 1)      # Velocidade
    v = zeros(nx + 1, ny + 1)      # Velocidade
    ω = zeros(nx + 1, ny + 1)      # Vorticidade

    u[1:nx+1, ny+1] .= 1   # Velocidade inicial da tampa

    # Realizando a montagem da matriz de Poisson
    A = matrizPoisson(nx, ny, δx, δy)
    # Vetor independente do sistema
    b = zeros((nx + 1) * (ny + 1))

    rx = 1 / (Re * δx * δx)
    ry = 1 / (Re * δy * δy)

    for iterationNumber in 1:nt
        ω = calculoContorno!(δx, δy, ψ, ω)
        ω = calculoVetorIndependente!(rx, ry, δx, δy, δt, ω, u, v, b)
        ψ = resolucaoSistemaLinear(nx, ny, b, A);
        uAnterior = copy(u)
        vAnterior = copy(v)
        u, v = atualizandoUeV(δx, δy, ψ, u, v)

        # Calculando resíduos em u e v
        residuoU = maximum(maximum(abs.(u - uAnterior)))
        residuoV = maximum(maximum(abs.(v - vAnterior)))
        # Printando informações do passo
        println("Passo: ", iterationNumber, "\tResíduo (u): ", residuoU, "\tResíduo (v): ", residuoV)

        # Se um dos erros for maior que 1e+8, aborta.
        # Se o erro de ambos forem menores que 1e-5, logo, convergiu.
        if (residuoU > 1e+8 || residuoV > 1e+8)
            println("Erro maior que 1e+8, abortando...")
            break
        elseif (residuoU < 1e-5 && residuoV < 1e-5)
            println("Convergiu!")
            return u, v
        end
    end
end

function matrizPoisson(nx::Int, ny::Int, δx, δy)
    # Definição da Matriz de Poisson (densa)
    A = zeros((nx + 1) * (ny + 1), (nx + 1) * (ny + 1))

    # Constantes, inverso do quadrado dos deltas
    Δx = 1 / (δx * δx)
    Δy = 1 / (δy * δy)

    # Constante
    z = -2 * (Δx + Δy)

    for i in 1:nx+1
        for j in 1:ny+1
            flatIndex = (i - 1) * (ny + 1) + j
            # Contorno -> Preenche diagonal principal com 1 (identidade)
            if (i == 1) || (i == nx + 1) || (j == 1) || (j == ny + 1)
                A[flatIndex, flatIndex] = 1
                # Fora do contorno -> Matriz pentadiagonal
            else
                # Elemento da esquerda
                A[flatIndex, (i-2)*(ny+1)+j] = Δx

                # Elemento do centro
                A[flatIndex, (i-1)*(ny+1)+j] = z

                # Elemento da direita
                A[flatIndex, (i)*(ny+1)+j] = Δx

                # Elemento de baixo
                A[flatIndex, (i-1)*(ny+1)+j-1] = Δy

                # Elemento de cima
                A[flatIndex, (i-1)*(ny+1)+j+1] = Δy
            end
        end
    end

    # Retorna matriz esparsa
    return SparseMatrixCSC(A)
end

function calculoContorno!(δx, δy, ψ, ω!)
    nx = size(ω!, 1) - 1
    ny = size(ω!, 2) - 1

    # Parede superior
    ω![1:nx+1, ny+1] = (-3 * δy .+ (7 / 2) * ψ[1:nx+1, ny+1] - 4 * ψ[1:nx+1, ny] + (1 / 2) * ψ[1:nx+1, ny-1]) / (δy^(2))

    # Parede inferior
    ω![1:nx+1, 1] = ((7 / 2) * ψ[1:nx+1, 1] - 4 * ψ[1:nx+1, 2] + (1 / 2) * ψ[1:nx+1, 3]) / (δy^(2))

    # Parede esquerda
    ω![1, 1:ny+1] = ((7 / 2) * ψ[1, 1:ny+1] - 4 * ψ[2, 1:ny+1] + (1 / 2) * ψ[3, 1:ny+1]) / (δx^(2))

    # Parede direita
    ω![nx+1, 1:ny+1] = ((7 / 2) * ψ[nx+1, 1:ny+1] - 4 * ψ[nx, 1:ny+1] + (1 / 2) * ψ[nx-1, 1:ny+1]) / (δx^(2))

    return ω!
end

function calculoVetorIndependente!(rx, ry, δx, δy, δt, ω, u, v, b!)
    nx = size(ω, 1) - 1
    ny = size(ω, 2) - 1

    # Atualizando ω com base nos valores anteriores
    ωNovo = copy(ω)
    ωNovo[2:nx, 2:ny] = ω[2:nx, 2:ny] + δt * (
        (u[2:nx, 2:ny] .* (ω[1:nx-1, 2:ny] - ω[3:nx+1, 2:ny]) ./ (2 * δx)) +
        (v[2:nx, 2:ny] .* (ω[2:nx, 1:ny-1] - ω[2:nx, 3:ny+1]) ./ (2 * δy)) +
        rx * (ω[3:nx+1, 2:ny] - 2 * ω[2:nx, 2:ny] + ω[1:nx-1, 2:ny]) +
        ry * (ω[2:nx, 3:ny+1] - 2 * ω[2:nx, 2:ny] + ω[2:nx, 1:ny-1])
    )

    for i in 2:nx
        for j in 2:ny
            flatIndex = (i - 1) * (ny + 1) + j
            b![flatIndex] = -ωNovo[i, j]
        end
    end

    return ωNovo
end

function resolucaoSistemaLinear(nx::Int, ny::Int, b, A)
    solucao = A \ b # Resolve o sistema linear 
    
    # Realizando reshape da solução, atribuindo à variável ψ
    # Transpose é necessário para que a matriz seja row-wise, ao invés de column-wise.
    ψ = transpose(reshape(solucao, (nx + 1, ny + 1)))
    return ψ;
end

function atualizandoUeV(δx, δy, ψ, u!, v!)
    nx = size(ψ, 1) - 1
    ny = size(ψ, 2) - 1

    # Atualizando u e v
    # Próximo do contorno, utilizando diferença centrada
    u![2, 2:ny] = (ψ[2, 3:ny+1] - ψ[2, 1:ny-1]) / (2 * δy)
    u![nx, 2:ny] = (ψ[nx, 3:ny+1] - ψ[nx, 1:ny-1]) / (2 * δy)
    u![2:nx, 2] = (ψ[2:nx, 3] - ψ[2:nx, 1]) / (2 * δy)
    u![2:nx, ny] = (ψ[2:nx, ny+1] - ψ[2:nx, ny-1]) / (2 * δy)
    v![2, 2:ny] = -(ψ[3, 2:ny] - ψ[1, 2:ny]) / (2 * δx)
    v![nx, 2:ny] = -(ψ[nx+1, 2:ny] - ψ[nx-1, 2:ny]) / (2 * δx)
    v![2:nx, 2] = -(ψ[3:nx+1, 2] - ψ[1:nx-1, 2]) / (2 * δx)
    v![2:nx, ny] = -(ψ[3:nx+1, ny] - ψ[1:nx-1, ny]) / (2 * δx)

    # Para os demais valores, utiliza diferença finita de quarta ordem
    u![3:nx-1, 3:ny-1] = (
        2 * ψ[3:nx-1, 1:ny-3] -
        16 * ψ[3:nx-1, 2:ny-2] +
        16 * ψ[3:nx-1, 4:ny] -
        2 * ψ[3:nx-1, 5:ny+1]
    ) / (24 * δy)

    v![3:nx-1, 3:ny-1] = -(
        2 * ψ[1:nx-3, 3:ny-1] -
        16 * ψ[2:nx-2, 3:ny-1] +
        16 * ψ[4:nx, 3:ny-1] -
        2 * ψ[5:nx+1, 3:ny-1]
    ) / (24 * δx)

    return u!, v!
end
