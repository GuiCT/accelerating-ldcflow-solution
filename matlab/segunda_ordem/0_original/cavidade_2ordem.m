clc
clear
format long
%clf

nx = 128;
ny = 128;
nt = 100000; 
Re = 100; 
dt = 0.01;
dx = 1/nx;
dy = 1/ny;
x = linspace(0,1,nx+1);
y = linspace(0,1,ny+1);
u0 = 1; % velocidade inicial da tampa

T  = 1;
%-------------------------------------------------------------------------

psi = zeros(nx+1,ny+1);  % Corrente
u = zeros(nx+1,ny+1);    % Velocidade
v = zeros(nx+1,ny+1);    % Velocidade
w = zeros(nx+1,ny+1);    % Vorticidade
u(1:nx+1,ny+1) = u0;     % Velocidade inicial da tampa

M = nx + 1;          % Malha computacional
Q = nx - 1;          % Malha computacional dos pontos internos do dom�nio



%% Calculando a matriz do sistema linear da equacao de poisson
tic
A = sparse((nx+1)*(ny+1),(nx+1)*(ny+1)); %Matriz do sistema linear de poisson
b = zeros((nx+1)*(ny+1),1); %Vetor independente do sistema linear de poisson
aux = 1;
cx = 1/(dx*dx);
cy = 1/(dy*dy);

for i = 1:nx+1
    for j = 1:ny+1
        %psi=0 no contorno
        if( (i==1) || (i==nx+1) || (j==1) || (j==ny+1) )
            A(aux,aux) = 1;
            b(aux) = 0;
            aux = aux+1;
            continue;
        end
        
        %Elemento da esquerda psi(i-1,j)
        A(aux,(i-2)*(ny+1)+j) = cx;
        
        %Elemento do centro psi(i,j)
        A(aux,(i-1)*(ny+1)+j) = -2*(cx + cy);
        
        %Elemento da direita psi(i+1,j)
        A(aux,(i)*(ny+1)+j) = cx;
        
        %Elemento de baixo psi(i,j-1)
        A(aux,(i-1)*(ny+1)+j-1) = cy;
        
        %Elemento de cima psi(i,j+1)
        A(aux,(i-1)*(ny+1)+j+1) = cy;
        
        aux = aux+1;
 
   end
end
%}

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
    aux = 1;
    for i = 1:nx+1
        for j = 1:ny+1
            %psi=0 no contorno
            if( (i==1) || (i==nx+1) || (j==1) || (j==ny+1) )
                b(aux) = 0;
                aux = aux+1;
                continue;
            end

            %Lado direito RHS
            b(aux) = - w(i,j);

            aux = aux+1;
        end
    end
  
    
    %Resolvendo o sistema linear
    solucao = A\b;    
% [solucao, r, k] = met_FOM(A, b);
% [solucao] = gauss(A,b);
    %Atribuindo a solucao na matriz psi
    aux = 1;
    for i = 1:nx+1
        for j = 1:ny+1
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
end 


%% Gr�ficos

% subplot(2, 2, 1);
% PlotPropriedade(u, nx, ny, 0, 0, 1, 1, 'Velocidade U');
% set(gcf, 'renderer', 'zbuffer');
% axis square;
% 
% subplot(2, 2, 2);
% PlotPropriedade(v, nx, ny, 0, 0, 1, 1, 'Velocidade V');
% set(gcf, 'renderer', 'zbuffer');
% axis square;
% 
% subplot(2, 2, 3);
% PlotPropriedade(psi, nx, ny, 0, 0, 1, 1, 'Corrente \psi');
% set(gcf, 'renderer', 'zbuffer');
% axis square;
% 
% figure
% PlotStreamlines(u, v, nx, ny, 0, 0, 1, 1, 50, 50);
% set(gcf, 'renderer', 'zbuffer');
% axis square;
% box on
toc
