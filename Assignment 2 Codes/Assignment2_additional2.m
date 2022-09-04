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

class1 = images(labels==1,:);
class2 = images(labels==2,:);
class3 = images(labels==3,:);
class4 = images(labels==4,:);
class5 = images(labels==5,:);
class6 = images(labels==6,:);
class7 = images(labels==7,:);
class8 = images(labels==8,:);
class9 = images(labels==9,:);
class0 = images(labels==0,:);

% class means
m1 = mean(class1);
m2 = mean(class2);
m3 = mean(class3);
m4 = mean(class4);
m5 = mean(class5);
m6 = mean(class6);
m7 = mean(class7);
m8 = mean(class8);
m9 = mean(class9);
m0 = mean(class0);
m = mean(images);
% class covariance matrix
s1 = cov(class1);
s2 = cov(class2);
s3 = cov(class3);
s4 = cov(class4);
s5 = cov(class5);
s6 = cov(class6);
s7 = cov(class7);
s8 = cov(class8);
s9 = cov(class9);
s0 = cov(class0);
% within class scatter matrix
sw = s1 + s2 +s3 +s4 +s5 +s6 +s7 +s8 +s9 +s0;

% between class scatter matrix
mb = zeros(10, 784);
mb(1, :) =  m1 - m;
mb(2, :) =  m2 - m;
mb(3, :) =  m3 - m;
mb(4, :) =  m4 - m;
mb(5, :) =  m5 - m;
mb(6, :) =  m6 - m;
mb(7, :) =  m7 - m;
mb(8, :) =  m8 - m;
mb(9, :) =  m9 - m;
mb(10, :) =  m0 - m;
sb = mb.' * mb;

% computing the LDA projection vector
[v, d] = eigs((inv(sw + 1e-7 * eye(784))) * sb);

% computing the projection score:
score = images * v(:, 1:2);

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
title('Dimension Reduction Using LDA', 'Interpreter', 'latex')
legend('Digit 1', 'Digit 2','Digit 3','Digit 4','Digit 5','Digit 6','Digit 7','Digit 8','Digit 9', 'Digit 0', 'Interpreter', 'latex');