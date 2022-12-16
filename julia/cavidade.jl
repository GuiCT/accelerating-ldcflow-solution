# Pacotes utilizados
using SparseArrays;
using Krylov;

function cavidade(nx::Int, ny::Int, nt::Int, Re::Int, T::Int)
    dx = 30/nx;
    dy = 1/ny;
    dt = T/nt;

    x = LinRange(0, 30, nx+1);
    y = LinRange(-0.5, 0.5, ny+1);
    
    u0 = y -> 24*y*(0.5 - y);
    w0 = y -> 48*y - 12;

    psi = zeros(nx+1, ny+1);   # Corrente
    u = zeros(nx+1, ny+1);     # Velocidade
    v = zeros(nx+1, ny+1);     # Velocidade
    w = zeros(nx+1, ny+1);     # Vorticidade
    u[1, (ny/2)+1:ny+1] = u0(y[(ny/2)+1:ny+1]);
    w[1, (ny/2)+1:ny+1] = w0(y[(ny/2)+1:ny+1]);

    # Utilizando função que gera matriz esparsa e vetor do Sistema Linear da Equação de Poisson
    A, b = matrizVectorPoisson(nx, ny);
    
    for _ in 1:nt
        # Atualizando w e psi
        w, psi = iterateOnce(Re, dx, dy, dt, u, v, w, psi, A, b);
        # Guardando valores de u e v
        u_old = copy(u);
        v_old = copy(v);
        # Atualizando u e v
        u, v = updateUandV(u, v, psi, dx, dy);
        # Calculando erros em u e v
        # Se um dos erros for maior que 1e+8, aborta.
        # Se o erro de ambos forem menores que 1e-5, logo, convergiu.
        if (maximum(abs.(u - u_old)) > 1e+8) || (maximum(abs.(v - v_old)) > 1e+8)
            println("Erro maior que 1e+8, abortando...");
            break;
        elseif (maximum(abs.(u - u_old)) < 1e-5) && (maximum(abs.(v - v_old)) < 1e-5)
            println("Convergiu!");
            break;
        end
    end
end

# Função que gera matriz esparsa do Sistema Linear da Equação de Poisson
function matrizVectorPoisson(nx::Int, ny::Int)
    # Inicializando matriz como uma matriz densa, será convertida posteriormente.
    A = zeros((nx+1)*(ny+1),(nx+1)*(ny+1));
    # Inicializando vetor independente
    b = zeros((nx+1)*(ny+1), 1);
    # Índice flat, incrementado a cada iteração
    flatIdx::Int = 1;
    # Constantes
    cx = 1/(dx*dx);
    cy = 1/(dy*dy);
    for i in 1:nx+1
        for j in 1:ny+1
            # No contorno, atribuindo psi = 0
            if (i == 1) || (i == nx+1) || (j == 1) || (j == ny+1)
                A[flatIdx, flatIdx] = 1;
                b[flatIdx] = 0;
            else
                A[flatIdx, (i-2)*(ny+1)+j] = cx;
                A[flatIdx, (i-1)*(ny+1)+j] = -2*(cx + cy);
                A[flatIdx, (i)*(ny+1)+j] = cx;
                A[flatIdx, (i-1)*(ny+1)+j-1] = cy;
                A[flatIdx, (i-1)*(ny+1)+j+1] = cy;
            end
            global flatIdx += 1;
        end
    end
    return A, b;
end

# Executa uma iteração do algoritmo original
function iterateOnce(Re::Int, dx::Float32, dy::Float32, dt::Float32, u::Array{Float32,2}, v::Array{Float32,2}, w::Array{Float32,2}, psi::Array{Float32,2}, A::SparseMatrixCSC{Float32,Int64}, b::Array{Float32,1})
    rx::Float32 = 1/(Re*dx*dx);
    ry::Float32 = 1/(Re*dy*dy);
    w_new::Array{Float32,2} = copy(w);
    w_new[2:nx, 2:ny] = w[2:nx, 2:ny] + dt*((u[2:nx, 2:ny] .* (w[1:nx-1, 2:ny] - w[3:nx+1, 2:ny])/(2*dx)) + (v[2:nx, 2:ny] .* (w[2:nx, 1:ny-1] - w[2:nx, 3:ny+1])/(2*dy)) + rx*(w[3:nx+1, 2:ny] - 2*w[2:nx, 2:ny] + w[1:nx-1, 2:ny]) + ry*(w[2:nx, 3:ny+1] - 2*w[2:nx, 2:ny] + w[2:nx, 1:ny-1]))
    
    # Solucionando a Equação de Poisson e Calculando PSI
    flatIdx::Int = 1;
    for i in 1:nx+1
        for j in 1:ny+1
            if (i == 1) || (i == nx+1) || (j == 1) || (j == ny+1)
                b[flatIdx] = 0;
            else
                b[flatIdx] = - new_w[i,j];
            end
            global flatIdx += 1;
        end
    end

    # Solucionando sistema linear esparso usando symmlq
    solucao = symmlq(A, b);
    # Criando vetor PSI utilizando reshape
    psi = reshape(solucao, (nx+1, ny+1));

    # Retornando novo w e psi.
    return w_new, psi;
end

function updateUandV(psi::Array{Float32,2}, u::Array{Float32,2}, v::Array{Float32,2}, dx::Float32, dy::Float32)
    u[2:nx, 2:ny] = 0.5 * (psi[2:nx, 3:ny+1] - psi[2:nx, 1:ny-1]) / dy
    v[2:nx, 2:ny] = -0.5 * (psi[3:nx+1, 2:ny] - psi[1:nx-1, 2:ny]) / dx
    return u, v;
end
