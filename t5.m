a3.clear()
load('mnist.mat')
A = cell2mat(img);
B = reshape(A,[60000,784])/255;
At = cell2mat(img_test);
Bt = reshape(At,[10000,784])/255;
subX = B(50000:end,:);
subY = labels(50000:end);
% r = 1000*rand(1000,2);
r=[];
bestPair = [];
regre = [0,100,500,1000,2000,5000,10000,20000,50000];

c = -5:2:15;
g = -15:2:3;
c = 2.^c;
g = 2.^g;

for i = 1:size(c,2)
    for j = 1: size(g,2)
        r = [r;c(i),g(j)];
    end
end

for i = 1:8
    trainX = B(regre(i)+1:regre(i+1),:);
    trainY = labels(regre(i)+1:regre(i+1));
    bestError = 99999999;
    for j = 1:size(r,1)
        params = templateSVM('KernelFunction', 'gaussian','BoxConstrain',r(j,1),'KernelScale',r(j,2));
        mdl = fitcecoc(trainX,trainY,'Learners', params,'Coding','onevsall');
        predSub = predict(mdl,subX);
        errors = sum(predSub ~= subY);
        if errors < bestError
            bestPair = [];
            bestPair = [bestPair;[r(j,1),r(j,2)]];
            bestError = errors;
        elseif errors == bestError
            bestPair = [bestPair;[r(j,1),r(j,2)]];
        end
    end
    r = bestPair;
end
% params = templateSVM('KernelFunction', 'gaussian');
% mdl = fitcecoc(subX,subY,'Learners', params,'Coding','onevsall');

py = predict(mdl,Bt);
error = sum(py ~= labels_test);
res = zeros(10,1);

for i=1:size(py,1)
    if py(i) ~= labels_test(i)
        res(labels_test(i)) = res(labels_test(i))+1;
    end
end