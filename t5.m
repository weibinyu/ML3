a3.clear()
load('mnist.mat')
A = cell2mat(img);
B = reshape(A,[60000,784])/255;
At = cell2mat(img_test);
Bt = reshape(At,[10000,784])/255;

subX = B(1:2000,:);
subY = labels(1:2000);


% params = templateSVM('KernelFunction', 'gaussian','BoxConstraint',0.0010499,'KernelScale',15.579);
% mdl = fitcecoc(subX,subY,'Learner',params,'Coding','onevsall');

h = hyperparameters('fitcecoc',B,labels,'SVM');
h(1,1).Optimize = false;
mdl = fitcecoc(subX,subY,'OptimizeHyperparameters',h,'Coding','onevsall');

py = predict(mdl,Bt);
error = sum(py ~= labels_test);
res = zeros(10,1);
for i=1:size(py,1)
    if py(i) ~= labels_test(i)
        res(labels_test(i)) = res(labels_test(i))+1;
    end
end