function [ solucao ] = Compact (A, N, f)
 
% ======================== Informa��es B�sicas ======================== %%
a = 0;
b = 1;
dx = (b-a)/N;       %Espa�amento na dire�ao x


%% ============================= Termo de b_0 ========================== %%

x1 = [a:dx:b];  %Vetor para fornecer os valores dos pontos discretos x_i 
y1 = x1;        %Vetor para fornecer os valores dos pontos discretos y_j

laplaciano = zeros((N-1));     %Matriz para armazenar o valor do laplaciano em cada ponto discreto da malha

for j = 2:N
    for i = 2:N
        
        %Substitui��o da vari�vel x na fun��o da equa��o de Poisson e constru��o do laplaciano
        laplaciano(i-1,j-1) = (f(i+1,j) + f(i-1,j) + 8*f(i,j) + f(i,j+1) + f(i,j-1))/(12);
%         laplaciano(i-1,j-1) = (f(i+1,j)+f(i-1,j)-4*f(i,j)+f(i,j+1)+f(i,j-1))/(12*f(i,j));
        
        %Constru��o do termo independente da equa��o de diferen�as finitas exponencial
%         term(i-1,j-1) = 6*((dx)^2)*(f(i,j))*exp(laplaciano(i-1,j-1));
        term(i-1,j-1) = 6*((dx)^2)*(laplaciano(i-1,j-1));
% 
    end
end
% laplaciano

%Transforma��o da matriz em vetor

for j = (N-1):-1:1
    
termb0((N-1)*(j-1)+1:(N-1)*j,1) = (term(j,:))';

end


%Constru��o do vetor formado pelas condi��es de contorno e pelo termo
%independente da f�rmula de diferen�as finitas exponencial

for i = 1:(N-1)^2;
%     R(i,1) = termb0(i)+B(i);
    R(i,1) = termb0(i);

 
end

solucao = A\R;


end