clear;
clc;

% Load the MNIST-1-5-8 dataset
load mnist-1-5-8.mat;

% transpose to make a picture in a row
images = images.';

% divide the dataset by different classes
class1 = images(labels==1,:);
class5 = images(labels==5,:);
class8 = images(labels==8,:);

% class means
m1 = mean(class1);
m5 = mean(class5);
m8 = mean(class8);
m = mean(images);
% class covariance matrix
s1 = cov(class1);
s5 = cov(class5);
s8 = cov(class8);
% within class scatter matrix
sw = s1 + s5 + s8;

% between class scatter matrix
mb = zeros(3, 784);
mb(1, :) =  m1 - m;
mb(2, :) =  m5 - m;
mb(3, :) =  m8 - m;
sb = mb.' * mb;

% computing the LDA projection vector
[v, d] = eigs((inv(sw + 1e-10 * eye(784))) * sb);

% computing the projection score:
score = images * v(:, 1:2);

figure(1);
hold on;
grid on;
k = find(labels==1);
plot(score(k, 1), score(k, 2), 'o');
k = find(labels==5);
plot(score(k, 1), score(k, 2), '*');
k = find(labels==8);
plot(score(k, 1), score(k, 2), '+');
title('Dimension Reduction Using LDA', 'Interpreter', 'latex')
legend('Digit 1', 'Digit 5', 'Digit 8', 'Interpreter', 'latex');

score = score * 100;

% ----- K-Means Clustering -----
[idx,C] = kmeans(score, 3);
x1 = min(score(:, 1)):0.01:max(score(:, 1));
x2 = min(score(:, 2)):0.01:max(score(:, 2));
[x1G,x2G] = meshgrid(x1, x2);
XGrid = [x1G(:), x2G(:)];
idx2Region = kmeans(XGrid, 3, 'MaxIter', 40, 'Start', C);
% ------------------------------

% Plot the results
figure(2);
hold on;
gscatter(score(:,1), score(:,2), idx);
plot(C(:,1), C(:,2), 'kx')
grid on;
title('Clustering Using K-Means', 'Interpreter', 'latex')
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster Centroid', 'Interpreter', 'latex');

figure(3);
% ----- Hierarchical Clustering -----
%Compute four clusters of the fisheriris data using single-linkage
Z = linkage(score(:,1:2),'average','euclidean'); %create the linkage tree using single-
%link
c = cluster(Z,'maxclust',3);
% -----------------------------------
% See how the cluster assignments correspond to the three species.
crosstab(c, labels)
% Create a dendrogram plot of Z, and visualize it.
dendrogram(Z)
title('Dendrogram Plot of Hierarchical Clustering', 'Interpreter', 'latex')

figure(4);
hold on;
gscatter(score(:,1), score(:,2), c);
grid on;
title('Clustering Using Hierarchical Clustering', 'Interpreter', 'latex')
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Interpreter', 'latex');

