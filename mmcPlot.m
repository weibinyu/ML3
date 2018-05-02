%% Maximal margin classifier
function beta = mmcPlot(X, y)
% Maximal margin classifier which computes and plots the hyperplane for 2D data

%% Solving the maximal margin classifier problem
p = 2;
H = eye(p+1);
H(1,1) = 0;
f = zeros(p+1,1);
b = -ones(size(X,1),1);
A = [-y -diag(y)*X];
beta = quadprog(H, f, A, b);

%% Find the support vectors
sv = (A*beta*(-1)<1.0001);

%% Normalizing beta
beta = beta/sqrt(beta(2:end)'*beta(2:end));

%% Find the distance from the support vectors to the hyperplane
% M should be the perpendicular distance from any support vector to the
% hyperplane
i=1;
while sv(i) ~= 1
    i=i+1;
end

M = y(i)'*(beta(1)+X(i,:)*beta(2:3));

% --------------------------------------------
% --------------------------------------------
% --------------------------------------------

d = M/sin(acos(beta(2)/sqrt(beta(2:end)'*beta(2:end))));

%% Plotting the MMC, its slab and support vectors
l = linspace(min(X(:,1)),max(X(:,1)),100);
pos = (y==1);
neg = (y==-1);

clf;
hold on;
% Plotting the data
plot(X(pos,1), X(pos,2),'ko','MarkerFaceColor','y');
plot(X(neg,1), X(neg,2),'ko','MarkerFaceColor','r');

% Plotting the MMC
plot(l, -beta(2)/beta(3)*l + -beta(1)/beta(3)); 

% Plotting one slab
plot(l,  -beta(2)/beta(3)*l + -(beta(1))/beta(3) + d,'m--');

% Plotting the support vectors
plot(X(sv,1), X(sv,2), 'ko','MarkerSize',10);

% Plotting the other slab
plot(l,  -beta(2)/beta(3)*l + -(beta(1))/beta(3) - d,'m--');

legend('Positive','Negative','Model 1', 'Slab', 'Support vectors');
xlim([-3 3])
ylim([-3 3])
hold off;

end

