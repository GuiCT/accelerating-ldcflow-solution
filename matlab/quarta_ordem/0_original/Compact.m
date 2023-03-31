function [ solucao ] = Compact (A, N, f)
 
% ======================== Informações Básicas ======================== %%
a = 0;
b = 1;
dx = (b-a)/N;       %Espaçamento na direçao x


%% ============================= Termo de b_0 ========================== %%

x1 = [a:dx:b];  %Vetor para fornecer os valores dos pontos discretos x_i 
y1 = x1;        %Vetor para fornecer os valores dos pontos discretos y_j

laplaciano = zeros((N-1));     %Matriz para armazenar o valor do laplaciano em cada ponto discreto da malha

for j = 2:N
    for i = 2:N
        
        %Substituição da variável x na função da equação de Poisson e construção do laplaciano
        laplaciano(i-1,j-1) = (f(i+1,j) + f(i-1,j) + 8*f(i,j) + f(i,j+1) + f(i,j-1))/(12);
%         laplaciano(i-1,j-1) = (f(i+1,j)+f(i-1,j)-4*f(i,j)+f(i,j+1)+f(i,j-1))/(12*f(i,j));
        
        %Construção do termo independente da equação de diferenças finitas exponencial
%         term(i-1,j-1) = 6*((dx)^2)*(f(i,j))*exp(laplaciano(i-1,j-1));
        term(i-1,j-1) = 6*((dx)^2)*(laplaciano(i-1,j-1));
% 
    end
end
% laplaciano

%Transformação da matriz em vetor

for j = (N-1):-1:1
    
termb0((N-1)*(j-1)+1:(N-1)*j,1) = (term(j,:))';

end


%Construção do vetor formado pelas condições de contorno e pelo termo
%independente da fórmula de diferenças finitas exponencial

for i = 1:(N-1)^2;
%     R(i,1) = termb0(i)+B(i);
    R(i,1) = termb0(i);

 
end

solucao = A\R;


end