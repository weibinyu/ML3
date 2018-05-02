a3.clear()
load('BM.mat')
trainX = batmanX(1:1000,:);
trainy = batmany(1:1000,:);
h = hyperparameters('fitcsvm',batmanX,batmany);
svm = fitcsvm(trainX,trainy,'OptimizeHyperparameters',h,'KernelFunction', 'gaussian');
testX = batmanX(1000:1100,:);
testy = batmanX(1000:1100,:);

y = predict(svm,batmanX);
error = sum(y ~= batmany);
errorR = error/1000000; %% error rate

a3.drawDB(batmanX,batmany,svm);