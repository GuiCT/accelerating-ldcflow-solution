function [results, performanceInfos] = cavidade_2ordem_function(nx, ny, Re, dt, nt, xRange, yRange)
%cavidade_2ordem_function.m Função que resolve o problema da cavidade com tampa móvel
%utilizando aproximações de segunda ordem.
% Inputs:
%   nx: Número de elementos em x (escalar)
%   ny: Número de elementos em y (escalar)
%   Re: Número de Reynolds (escalar)
%   dt: Passo de integração temporal (escalar) (Opcional, por padrão 0.001)
%   nt: Número (MÁXIMO) de iterações (escalar) (Opcional, por padrão 10000)
%   xRange: Intervalo do espaço linear em x (2,1) (Opcional, por padrão [0;1])
%   yRange: Intervalo do espaço linear em y (2,1) (Opcional, por padrão [0;1])

dx = 1/nx;
dy = 1/ny;

if exist("nt", "var") == 0
    nt = 10000;
end

if exist("dt", "var") == 0
    dt = 0.001;
end

if exist("xRange", "var") == 0
    xRange = [0; 1];
end

if exist("yRange", "var") == 0
    yRange = [0; 1];
end

x = linspace(xRange(1), xRange(2), nx+1);
y = linspace(yRange(1), yRange(2), ny+1);

psi = zeros(nx+1, ny+1);    % Corrente
u = zeros(nx+1, ny+1);      % Velocidade
v = zeros(nx+1, ny+1);      % Velocidade
w = zeros(nx+1, ny+1);      % Vorticidade
u(1:nx+1, ny+1) = 1;        % Velocidade inicial da tampa

tic % Começa a contagem do tempo de execução
A = matrizPoisson(nx, ny, dx, dy); % Matriz do sistema linear

rx = 1/(Re*dx*dx);
ry = 1/(Re*dy*dy);
for iterationNumber = 1:nt
    w = calculoContorno(nx, ny, dx, dy, psi, w);
    [w, b] = calculoVetorIndependente(nx, ny, rx, ry, dx, dy, dt, w, u, v);
    psi = resolucaoSistemaLinear(nx, ny, b, A);

    u_velho = u;
    v_velho = v;

    % Atualizando u e v
    % Próximo do contorno, utilizando diferença centrada
    u(2, 2:ny) = (0.5/dy)*(psi(2, 3:ny+1) - psi(2, 1:ny-1));
    u(nx, 2:ny) = (0.5/dy)*(psi(nx, 3:ny+1) - psi(nx, 1:ny-1));
    u(2:nx, 2) = (0.5/dy)*(psi(2:nx, 3) - psi(2:nx, 1));
    u(2:nx, ny) = (0.5/dy)*(psi(2:nx, ny+1) - psi(2:nx, ny-1));

    v(2, 2:ny) = (-0.5/dx)*(psi(3, 2:ny) - psi(1, 2:ny));
    v(nx, 2:ny) = (-0.5/dx)*(psi(nx+1, 2:ny) - psi(nx-1, 2:ny));
    v(2:nx, 2) = (-0.5/dx)*(psi(3:nx+1, 2) - psi(1:nx-1, 2));
    v(2:nx, ny) = (-0.5/dx)*(psi(3:nx+1, ny) - psi(1:nx-1, ny));
    % Para os demais valores, utiliza diferença finita de quarta ordem
    u(3:nx-1, 3:ny-1) = (0.041666666666667/dy)*( ...
        2*psi(3:nx-1, 1:ny-3) - ...
        16*psi(3:nx-1, 2:ny-2) + ...
        16*psi(3:nx-1, 4:ny) - ...
        2*psi(3:nx-1, 5:ny+1) ...
        );

    v(3:nx-1, 3:ny-1) = (-0.041666666666667/dy)*( ...
        2*psi(1:nx-3, 3:ny-1) - ...
        16*psi(2:nx-2 ,3:ny-1) + ...
        16*psi(4:nx ,3:ny-1) - ...
        2*psi(5:nx+1 ,3:ny-1) ...
        );

    erro_maximo_u = max(max(abs(u_velho - u)));
    erro_maximo_v = max(max(abs(v_velho - v)));

    fprintf('Passo: %d     Residuo:%.10e    Residuo:%.10e\n', iterationNumber, erro_maximo_u, erro_maximo_v);

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

function w = calculoContorno(nx, ny, dx, dy, psi, w)
% Parede superior
w(1:nx+1, ny+1) = (-3*dy + (7/2)*psi(1:nx+1,ny+1) - 4*psi(1:nx+1,ny) + (1/2)*psi(1:nx+1,ny-1))/(dy^(2));

% Parede inferior
w(1:nx+1,1) = ((7/2)*psi(1:nx+1,1) - 4*psi(1:nx+1,2) + (1/2)*psi(1:nx+1,3))/(dy^(2));

% Parede esquerda
w(1,1:ny+1) = ((7/2)*psi(1,1:ny+1) - 4*psi(2,1:ny+1) + (1/2)*psi(3,1:ny+1))/(dx^(2));

% Parede direita
w(nx+1,1:ny+1) = ((7/2)*psi(nx+1,1:ny+1) - 4*psi(nx,1:ny+1) + (1/2)*psi(nx-1,1:ny+1))/(dx^(2));
end

function [wNovo, b] = calculoVetorIndependente(nx, ny, rx, ry, dx, dy, dt, w, u, v)
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

flatIndex = nx+3;
for i = 2:nx
    for j = 2:ny
        % Lado direito RHS
        b(flatIndex) = - wNovo(i, j);
        flatIndex = flatIndex+1;
    end
    flatIndex = flatIndex+2;
end
end

function psi = resolucaoSistemaLinear(nx, ny, b, A)
solucao = A\b; % Resolve o sistema linear
% Utilizando reshape para atribuir a solução à psi
% Transposta para ser row-wise
psi = reshape(solucao, [nx+1, ny+1])';
end