a3.clear()
load('mnist.mat')
A = cell2mat(img);
B = reshape(A,[60000,784])/255;
At = cell2mat(img_test);
Bt = reshape(At,[10000,784])/255;

subX = B(1:2000,:);
subY = labels(1:2000);

r = 1000*rand(2,2);
L = zeros(2,1);

params = templateSVM('KernelFunction', 'polynomial');
mdl = fitcecoc(subX,subY,'Learners', params,'Coding','onevsall');

py = predict(mdl,Bt);
error = sum(py ~= labels_test);
res = zeros(10,1);
for i=1:size(py,1)
    if py(i) ~= labels_test(i)
        res(labels_test(i)) = res(labels_test(i))+1;
    end
end