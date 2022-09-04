clear;
clc;

load Train.mat;
images = zeros(2000, 784);
labels = zeros(2000, 1);
for i = 1:30:59971
    labels(i, 1) = Train{i, 2};
    images(i, :) = reshape(Train{i, 1},1,[]);
    disp(i)
end

% Normalize the data (mean or zscore)
images_nor = images - mean(images);
% images_nor = zscore(images);
% images_nor = images;
% Calculate the covariance matrix
r = cov(images_nor);
% Eigenvalue decomposition
[v, d] = eigs(r);
% Calculate scores
score = images_nor * v(:,1:2);
% ------------------------------------

% Plot the results
figure(1);
hold on;
grid on;
k = find(labels==1);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==2);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==3);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==4);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==5);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==6);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==7);
plot(score(k, 1), score(k, 2), '.');
k = find(labels==8);
plot(score(k, 1), score(k, 2), 'r.');
k = find(labels==9);
plot(score(k, 1), score(k, 2), 'b.');
k = find(labels==0);
plot(score(k, 1), score(k, 2), 'c.');
title('Dimension Reduction Using PCA', 'Interpreter', 'latex')
legend('Digit 1', 'Digit 2','Digit 3','Digit 4','Digit 5','Digit 6','Digit 7','Digit 8','Digit 9', 'Digit 0', 'Interpreter', 'latex');