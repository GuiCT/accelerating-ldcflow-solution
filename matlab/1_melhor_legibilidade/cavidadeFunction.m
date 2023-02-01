function [results, performanceInfos] = cavidadeFunction(nx, ny, nt, Re, T)
dx = 30/nx;
dy = 1/ny;
dt = T/nt;

x = linspace(0, 30, nx+1);
y = linspace(-0.5, 0.5, ny+1);

u0 = @(y) 24*y .* (0.5 - y);  % Velocidade inicial da tampa
w0 = @(y) 48*y - 12;

u = zeros(nx+1, ny+1);    % Velocidade
v = zeros(nx+1, ny+1);    % Velocidade
w = zeros(nx+1, ny+1);    % Vorticidade
u(1, (ny/2)+1:ny+1) = u0(y((ny/2)+1:ny+1));
w(1, (ny/2)+1:ny+1) = w0(y((ny/2)+1:ny+1));

tic % Começa a contagem do tempo de execução
A = matrizPoisson(nx, ny, dx, dy); % Matriz do sistema linear

rx = 1/(Re*dx*dx);
ry = 1/(Re*dy*dy);
for iterationNumber = 1:nt
    [w, b] = calculoVetorIndependente(w, u, v, nx, ny, rx, ry, dx, dy, dt);
    psi = resolucaoSistemaLinear(A, b, nx, ny);

    u_velho = u;
    v_velho = v;

    %Atualizando u e v
    u(2:nx, 2:ny) = (0.5/dy)*(psi(2:nx, 3:ny+1) - psi(2:nx, 1:ny-1));
    v(2:nx, 2:ny) = (-0.5/dx)*(psi(3:nx+1, 2:ny) - psi(1:nx-1, 2:ny));

    erro_maximo_u = max(max(abs(u_velho - u)));
    erro_maximo_v = max(max(abs(v_velho - v)));

    if erro_maximo_u>1e+8 || erro_maximo_v>1e+8
        performanceInfos.convergence = false;
        break;
    elseif erro_maximo_u<1e-5 && erro_maximo_v<1e-5
        performanceInfos.convergence = true;
        performanceInfos.nIterations = iterationNumber;
        break;
    end
end
performanceInfos.tElapsed = toc;

results.x = x;
results.y = y;
results.u = u;
results.v = v;
end

function A = matrizPoisson(nx, ny, dx, dy)
A = zeros((nx+1)*(ny+1), (nx+1)*(ny+1)); % Matriz do sistema linear
% Inicializa como matriz densa para evitar problemas com indexing esparso
% Ocupa mais memória temporariamente
flatIndex = 1;
cx = 1/(dx*dx);
cy = 1/(dy*dy);

for i = 1:nx+1
    for j = 1:ny+1
        % Contorno
        if ((i==1) || (i==nx+1) || (j==1) || (j==ny+1))
            A(flatIndex,flatIndex) = 1;
        else
            %Elemento da esquerda psi(i-1, j)
            A(flatIndex, (i-2)*(ny+1)+j) = cx;

            %Elemento do centro psi(i, j)
            A(flatIndex, (i-1)*(ny+1)+j) = -2*(cx + cy);

            %Elemento da direita psi(i+1, j)
            A(flatIndex, (i)*(ny+1)+j) = cx;

            %Elemento de baixo psi(i, j-1)
            A(flatIndex, (i-1)*(ny+1)+j-1) = cy;

            %Elemento de cima psi(i, j+1)
            A(flatIndex, (i-1)*(ny+1)+j+1) = cy;
        end
        flatIndex = flatIndex+1;
    end
end

% Transforma A em matriz esparsa
A = sparse(A);
end

function [wNovo, b] = calculoVetorIndependente(w, u, v, nx, ny, rx, ry, dx, dy, dt)
% Atualizando w com base no w anterior
wTemp = w;
wTemp(2:nx, 2:ny) = w(2:nx, 2:ny) + dt* ...
( ...
    (u(2:nx, 2:ny).*(w(1:nx-1, 2:ny) - w(3:nx+1, 2:ny))./(2*dx)) + ...
    (v(2:nx, 2:ny).*(w(2:nx,1:ny-1) - w(2:nx, 3:ny+1))./(2*dy)) + ...
    rx*(w(3:nx+1, 2:ny) - 2*w(2:nx, 2:ny) + w(1:nx-1, 2:ny)) + ...
    ry*(w(2:nx, 3:ny+1) - 2*w(2:nx, 2:ny) + w(2:nx, 1:ny-1)) ...
);
wNovo = wTemp;

% Montando vetor independente
b = zeros((nx + 1)*(ny + 1), 1);

flatIndex = 1;
for i = 1:nx+1
    for j = 1:ny+1
        % Se está no contorno
        if ((i == 1) || (i == nx+1) || (j == 1) || (j == ny+1))
        else
            % Lado direito RHS
            b(flatIndex) = - wNovo(i, j);
        end
        flatIndex = flatIndex+1;
    end
end
end

function psi = resolucaoSistemaLinear(A, b, nx, ny)
solucao = A\b; % Resolve o sistema linear
% Utilizando reshape para atribuir a solução à psi
% Transposta para ser row-wise
psi = reshape(solucao, [nx+1, ny+1])';
end