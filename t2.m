a3.clear()
load('titanic.mat')

y = titanic(:,1);
X = titanic(:,2:8);
trainX = X(1:600,:);
testX = X(601:end,:);
trainY = y(1:600,:);
testY = y(601:end,:);

tree = fitctree(trainX,trainY,'Prune','on');
[E,SE,Nleaf,BestLevel] = cvloss(tree,'SubTree','All'); %%find best level to find best alpha
treeP = prune(tree,'Alpha',tree.PruneAlpha(BestLevel+1));
view(treeP,'Mode','graph');