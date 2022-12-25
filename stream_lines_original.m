function PlotStreamlines(U, V, nx, ny, xmin, ymin, xmax, ymax, SeedsX, SeedsY)
    dx = (xmax-xmin)/nx;
    dy = (ymax-ymin)/ny;

    x = xmin;
    for i=1:nx+1

        y = ymin;
        for j=1:ny+1
            pontosMalhaX(i) = x;
            pontosMalhaY(j) = y;

           

            y = y + dx;
        end
        x = x+dx;
    end
    
    %Criando os pontos seed pras streamlines
    dx = (xmax-xmin)/(SeedsX-1);
    dy = (ymax-ymin)/(SeedsY-1);
    
    indice = 1;
    x = xmin;
    y = ymin;
    for i=1:SeedsX
        %y = ymin;
        %for j=1:SeedsY
            seedsX(indice) = x;
            
            seedsY(indice) = y;
            
            seedsX(indice+1) = x;
            seedsY(indice+1) = 1-y;
            indice = indice + 2;
            
            y = y + dy;
        %end
        x = x + dx;
    end
        
    hlines=streamline(pontosMalhaX, pontosMalhaY, U', V', seedsX, seedsY);
    set(hlines, 'Color', 'k')
    axis([xmin xmax ymin ymax]);
    set(gca, 'Box', 'on');
    %axis square;
    xlabel('x');
    ylabel('y');
    %title('Linhas de Corrente');
end