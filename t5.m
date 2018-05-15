a3.clear()
load('mnist.mat')
A = cell2mat(img);
B = reshape(A,[60000,784])/255;
At = cell2mat(img_test);
Bt = reshape(At,[10000,784])/255;
subX = B(59900:end,:);
subY = labels(59900:end);

regre = [0,100,1000,10000,50000];

% r = 1000*rand(1000,2);
% c = -5:2:15;
% g = -15:2:3;
% c = 2.^c;
% g = 2.^g;
%  

s=linspace(0.001,10,100)';
m=linspace(0.001,10000,100)';
r=[];
% m=randperm(1000);
% x=zeros(1,1000);
for i = 1:size(s,1)
     for j = 1: size(s,1)         
         r = [r;s(i),m(j)];
    end
end


for i = 1:4
    trainX = B(regre(i)+1:regre(i+1),:);
    trainY = labels(regre(i)+1:regre(i+1));
    bestError = 99999999;
    for j = 1:size(r,1)
        disp(+j);
        params = templateSVM('KernelFunction', 'gaussian','KernelScale',r(j,1),'BoxConstrain',r(j,2));
        mdl = fitcecoc(trainX,trainY,'Learners', params,'Coding','onevsall');
        errors = sum(predict(mdl,subX) ~= subY);
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

py = predict(mdl,Bt);
error = sum(py ~= labels_test);
res = zeros(10,1);

for i=1:size(py,1)
    if py(i) ~= labels_test(i)
        res(labels_test(i)) = res(labels_test(i))+1;
    end
end