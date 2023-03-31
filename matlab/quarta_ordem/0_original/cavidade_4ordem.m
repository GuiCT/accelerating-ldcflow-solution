clc
clear
format long

nx = 64;
ny = 64;
nt = intmax; 
Re = 1000; 
dt = 0.001;
dx = 1/nx;
dy = 1/ny;
x = linspace(0,1,nx+1);
y = linspace(0,1,ny+1);
u0 = 1;                     %velocidade inicial da tampa

%-------------------------------------------------------------------------

psi = zeros(nx+1,ny+1); 
u = zeros(nx+1,ny+1); 
v = zeros(nx+1,ny+1);
w = zeros(nx+1,ny+1);
u(1:nx+1,ny+1) = u0;

%% Calculando a matriz do sistema linear da equacao de poisson
%
tic
%Construção dos vetores para armazenar as diagonais
N = nx;

for i = 1:(N-1)^2
    A20(i) = -20;                   %Atribuição dos valores de cada componente do vetor
end
 
for i = 1:(((N-1)-1)/(N-1))*((N-1)^2)
    A4(i) = 4;                      %Atribuição dos valores de cada componente do vetor
end
 
for i = 1:(N-1)^2-1
    if mod(i,(N-1)) == 0            %Condição para estabelecer a posição do componente 0
         A40(i) = 0; 
    else A40(i) = 4;                %Condição para estabelecer a posição do componente 4
    end
end

for i = 1:(((N-1)-1)/(N-1))*((N-1)^2)-1
    if mod(i,(N-1)) == 0            %Condição para estabelecer a posição do componente 0
         A10(i) = 0; 
    else A10(i) = 1;                %Condição para estabelecer a posição do componente 1
    end
end
 
for i = 1:(((N-1)-1)/(N-1))*((N-1)^2)+1
    if mod(i,(N-1)) == 1            %Condição para estabelecer a posição do componente 0
         A01(i) = 0; 
    else A01(i) = 1;                %Condição para estabelecer a posição do componente 1
    end
end
 
%% ========= Matriz A com os coeficientes do sistema linear ============ %%
A = diag(A10,(-(N-1)-1)) + diag(A4,(-(N-1))) + diag(A01,(-(N-1)+1)) + diag(A40,(-1)) + diag(A20) + diag(A40,1) + diag(A01,((N-1)-1))...
    + diag(A4,((N-1))) + diag(A10,((N-1)+1));
A = sparse(A);
% A = decomposition(A, "auto");

for n = 1:nt
   
    %% Calculando w nos contornos
    
    %Parede superior
    j = ny+1;
    for i = 1:nx+1
        w(i,j) = (-3*dy + (7/2)*psi(i,j) - 4*psi(i,j-1) + (1/2)*psi(i,j-2))/(dy^(2));
    end

    %Parede inferior
    j = 1;
    for i = 1:nx+1
        w(i,j) = ((7/2)*psi(i,j) - 4*psi(i,j+1) + (1/2)*psi(i,j+2))/(dy^(2));
    end

    %Parede esquerda
    i = 1;
    for j = 1:ny+1
    	w(i,j) = ((7/2)*psi(i,j) - 4*psi(i+1,j) + (1/2)*psi(i+2,j))/(dx^(2));
    end

    %Parede direita
    i = nx+1;
    for j = 1:ny+1
    	w(i,j) = ((7/2)*psi(i,j) - 4*psi(i-1,j) + (1/2)*psi(i-2,j))/(dx^(2));
    end

    %% RESOLVENDO A EQUACAO DE TRANSPORTE EXPLICITAMENTE E CALCULANDO W
    
            rx = 1/(Re*dx*dx);
            ry = 1/(Re*dy*dy);
            wnovo = w;
            for i = 2:nx 
                for j = 2:ny
                    wnovo(i,j) = w(i,j) + dt*( (u(i,j)*(w(i-1,j)-w(i+1,j))/(2*dx)) + (v(i,j)*(w(i,j-1)-w(i,j+1))/(2*dy)) + rx*(w(i+1,j)-2*w(i,j)+w(i-1,j)) + ry*(w(i,j+1)-2*w(i,j)+w(i,j-1))    );
                end
            end
            w = wnovo;


    %% RESOLVENDO A EQUACAO DE POISSON E CALCULANDO PSI
    
    %Resolvendo o sistema linear
    solucao = Compact (A, nx, -w);
%     solucao = A\b;
    %Atribuindo a solucao na matriz psi
    aux = 1;
    for i = 2:nx
        for j = 2:ny
            psi(i,j) = solucao(aux);
            aux = aux + 1;
        end
    end
    
    %% Atualizando os valores de u e v
    
    uvelho = u;
    vvelho = v;
     
    %Atualizando u
    for i = 2:nx
        for j = 2:ny
            u(i,j) = 0.5*(psi(i,j+1)-psi(i,j-1))/dy;
        end
    end

    %Atualizando v
    for i= 2:nx
        for j = 2:ny
            v(i,j) = -0.5*(psi(i+1,j)-psi(i-1,j))/dx;
        end
    end
    
    errou = max(max(abs(uvelho-u)));
    errov = max(max(abs(vvelho-v)));
    fprintf('Passo: %d     Residuo:%.10e    Residuo:%.10e\n', n, errou, errov);
    
    if errou>1e+8 || errov>1e+8
    	disp('A solucao nao convergiu :(');
        break;
    elseif errou<1e-5 && errov<1e-5 
    	disp('A solucao convergiu :D');
    	break;
    end
    
    
% PlotPropriedade(psi, nx, ny, 0, 0, 1, 1);
% set(gcf, 'renderer', 'zbuffer');
% axis square;
end 

toc
%% Gráficos
%{
figure('Color',[1 1 1]);
PlotPropriedade(u, nx, ny, 0, 0, 1, 1, 'Velocidade U');
set(gcf, 'renderer', 'zbuffer');
axis square;

figure('Color',[1 1 1]);
PlotPropriedade(v, nx, ny, 0, 0, 1, 1, 'Velocidade V');
set(gcf, 'renderer', 'zbuffer');
axis square;

figure('Color',[1 1 1]);
PlotPropriedade(psi, nx, ny, 0, 0, 1, 1, 'Corrente \psi');
set(gcf, 'renderer', 'zbuffer');
axis square;

figure('Color',[1 1 1]);
PlotStreamlines(u, v, nx, ny, 0, 0, 1, 1, 50, 50);
set(gcf, 'renderer', 'zbuffer');
axis square;
box on

%}

% figure('Color',[1 1 1]);
% quiver(x,y,u',v')
% xlim([0 1]); ylim([0 1.02]);
