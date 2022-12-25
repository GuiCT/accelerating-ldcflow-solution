# Pacotes utilizados
using SparseArrays;

function cavidade(nx::Int, ny::Int, nt::Int, Re::Int, T::Int)
    dx = 30/nx;
    dy = 1/ny;
    dt = T/nt;

    x = LinRange(0, 30, nx+1);
    y = LinRange(-0.5, 0.5, ny+1);
    
    u0(t) = 24*t .* (0.5 .- t);
    w0(t) = 48*t .- 12;

    psi = zeros(nx+1, ny+1);   # Corrente
    u = zeros(nx+1, ny+1);     # Velocidade
    v = zeros(nx+1, ny+1);     # Velocidade
    w = zeros(nx+1, ny+1);     # Vorticidade
    u[1, (ny÷2)+1:ny+1] = u0(y[(ny÷2)+1:ny+1]);
    w[1, (ny÷2)+1:ny+1] = w0(y[(ny÷2)+1:ny+1]);

    # Utilizando função que gera matriz esparsa do Sistema Linear da Equação de Poisson
    A = matrizPoisson(dx, dy, nx, ny);
    # Vetor independente
    b = zeros((nx+1)*(ny+1));
    
    for p in 1:nt
        # Obtendo psi e novo w, b é alterado pela função.
        psi, w = iterateOnce!(dx, dy, dt, nx, ny, Re, u, v, A, w, b);

        # Guardando valores de u e v
        u_old = copy(u);
        v_old = copy(v);

        # Atualizando u e v
        updateUandV!(dx, dy, nx, ny, psi, u, v);

        # Calculando erros em u e v
        u_error = maximum(maximum(abs.(u - u_old)));
        v_error = maximum(maximum(abs.(v - v_old)));
        # Printando informações do passo
        println("Passo: ", p, " Erro em u: ", u_error, " Erro em v: ", v_error, "")

        # Se um dos erros for maior que 1e+8, aborta.
        # Se o erro de ambos forem menores que 1e-5, logo, convergiu.
        if (u_error > 1e+8 || v_error > 1e+8)
            println("Erro maior que 1e+8, abortando...");
            break;
        elseif (u_error < 1e-5 && v_error < 1e-5)
            println("Convergiu!");
            break;
        end
    end

    return u, v;
end

# Função que gera matriz esparsa do Sistema Linear da Equação de Poisson
function matrizPoisson(dx, dy, nx::Int, ny::Int)
    # Inicializando matriz como uma matriz densa, será convertida posteriormente.
    A = zeros((nx+1)*(ny+1),(nx+1)*(ny+1));

    # Constantes
    cx = 1/(dx*dx);
    cy = 1/(dy*dy);

    for i in 1:nx+1
        for j in 1:ny+1
            # Índice flat
            flatIdx = (i-1)*(ny+1)+j;
            # No contorno, atribuindo psi = 0
            if (i == 1) || (i == nx+1) || (j == 1) || (j == ny+1)
                A[flatIdx, flatIdx] = 1;
            else
                A[flatIdx, (i-2)*(ny+1)+j] = cx;
                A[flatIdx, (i-1)*(ny+1)+j] = -2*(cx + cy);
                A[flatIdx, (i)*(ny+1)+j] = cx;
                A[flatIdx, (i-1)*(ny+1)+j-1] = cy;
                A[flatIdx, (i-1)*(ny+1)+j+1] = cy;
            end
        end
    end

    return SparseMatrixCSC(A);
end

# Executa uma iteração do algoritmo original
function iterateOnce!(dx, dy, dt, nx::Int, ny::Int, Re::Int, u, v, A, w, b!)
    rx = 1/(Re*dx*dx);
    ry = 1/(Re*dy*dy);
    w_new = copy(w);
    w_new[2:nx, 2:ny] = w[2:nx, 2:ny] + dt*((u[2:nx, 2:ny] .* (w[1:nx-1, 2:ny] - w[3:nx+1, 2:ny])/(2*dx)) + (v[2:nx, 2:ny] .* (w[2:nx, 1:ny-1] - w[2:nx, 3:ny+1])/(2*dy)) + rx*(w[3:nx+1, 2:ny] - 2*w[2:nx, 2:ny] + w[1:nx-1, 2:ny]) + ry*(w[2:nx, 3:ny+1] - 2*w[2:nx, 2:ny] + w[2:nx, 1:ny-1]))


    # Solucionando a Equação de Poisson e Calculando PSI
    for i in 1:nx+1
        for j in 1:ny+1
            flatIdx = (i-1)*(ny+1)+j;
            if (i == 1) || (i == nx+1) || (j == 1) || (j == ny+1)
                b![flatIdx] = 0;
            else
                b![flatIdx] = - w_new[i,j];
            end
        end
    end

    # Solucionando sistema linear esparso por método direto
    solucao = A\b!;
    # Criando vetor PSI utilizando reshape
    psi = transpose(reshape(solucao, (nx+1, ny+1)));

    # Retornando novo w e psi.
    return psi, w_new;
end

function updateUandV!(dx, dy, nx::Int, ny::Int, psi, u!, v!)
    u![2:nx, 2:ny] = 0.5 * (psi[2:nx, 3:ny+1] - psi[2:nx, 1:ny-1]) / dy;
    v![2:nx, 2:ny] = -0.5 * (psi[3:nx+1, 2:ny] - psi[1:nx-1, 2:ny]) / dx;
end

# cavidade(64, 64, 100000, 100, 100);