a3.clear()
load('data1.mat')

%% draw tree
mdl = fitctree(X,y);
view(mdl,'Mode','graph');

%% draw boundary
drawDB(X,y,mdl);

%% cross validate
cvmodel = crossval(mdl,'KFold',6);
L = kfoldLoss(cvmodel);

%% create 1 Miniparent
figure 
mTree = fitctree(X,y,'MinParentSize',1);
view(mTree,'Mode','graph')
drawDB(X,y,mTree)
cvmodel2 = crossval(mTree,'KFold',6);
L2 = kfoldLoss(cvmodel2);

%% fucntion draw boundary
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

%% 