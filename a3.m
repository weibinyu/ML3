classdef a3 % Same name as .m file
    properties % Not in use
    end
    methods(Static)
        function clear()
            clear; % Clear Command Window
            close all; % Close all figure windows
            clc % Clear Workspace
        end
        
        function g = sigmoid(z)
        g = 1./(1+exp(-z));
        end
        
        function beta = calcB(X,y)%calculate beta using X and y
        n = length(X);
        X2 = [ones(n,1),X];
        beta = ((X2.'*X2)^(-1))*(X2.')*y;
        end
        
        function norm = normalize(X)
        M = mean(X);
        S = std(X);
        XX = X;
        for i=1:size(X,1)
           XX(i,:) = (XX(i,:)-M)./S;
        end
            norm = XX;
        end
        
        function mse = J(X,y,B)
        sum = 0;
        for i=1:length(X)
            sum = sum+((B(1,1)+B(2,1)*X(i,1))-y(i,1))^2;
        end
        mse = sum/length(X);
        end
        
        %%fucntion draw boundary
        function drawDB(X,y,tree)
        n=60;
        x0=linspace(min(X(:,1)),max(X(:,1)),n);   
        xx0=linspace(min(X(:,2)),max(X(:,2)),n);
        [x1,xx1] = meshgrid(x0,xx0);
        grid = horzcat(x1(:), xx1(:), zeros(length(x1(:)), 1));
        A=zeros(n);
        for i=1: n
            for j=1: n
                z=[x0(i) xx0(j)];
                A(j,i)=predict(tree, z);
            end
        end
        gscatter(grid(:,1),grid(:,2),A(:));
        hold on
        gscatter(X(:,1),X(:,2),y);

        end

    end
end