a3.clear()
load('data1.mat')

%% draw tree
mdl = fitctree(X,y);
view(mdl,'Mode','graph');

%% draw boundary
a3.drawDB(X,y,mdl);

%% cross validate
cvmodel = crossval(mdl,'KFold',6);
L = kfoldLoss(cvmodel);

%% create 1 Minparent
figure 
mTree = fitctree(X,y,'MinParentSize',1);
view(mTree,'Mode','graph')
a3.drawDB(X,y,mTree)
cvmodel2 = crossval(mTree,'KFold',6);
L2 = kfoldLoss(cvmodel2);

