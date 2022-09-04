clear;
clc;

% Load the MNIST-1-5-8 dataset
load mnist-1-5-8.mat;

% Visualize one of the images
% i = unidrnd(600);
% im = reshape(images(:, i), [28, 28]);
% figure(1);
% imshow(imresize(im, 10),[]); 
% title('One of the images in MNIST-1-5-8 dataset', 'Interpreter', 'latex')

% --- Principle Component Analysis ---
% transpose to make a picture in a row
images = images.';
% Normalize the data (mean or zscore)
images_nor = images - mean(images);
% images_nor = zscore(images);
% Calculate the covariance matrix
r = cov(images_nor);
% Eigenvalue decomposition
[v, ~] = eigs(r);
% Calculate scores
score = images_nor * v(:,1:2);
% ------------------------------------

% Plot the results
figure(2);
hold on;
k = find(labels==1);
plot(score(k, 1), score(k, 2), 'o');
k = find(labels==5);
plot(score(k, 1), score(k, 2), '*');
k = find(labels==8);
plot(score(k, 1), score(k, 2), '+');
grid on;
title('Dimension Reduction Using PCA', 'Interpreter', 'latex')
legend('Digit 1', 'Digit 5', 'Digit 8', 'Interpreter', 'latex');

% ----- K-Means Clustering -----
[idx,C] = kmeans(score, 3);
x1 = min(score(:, 1)):0.01:max(score(:, 1));
x2 = min(score(:, 2)):0.01:max(score(:, 2));
[x1G,x2G] = meshgrid(x1, x2);
XGrid = [x1G(:), x2G(:)];
idx2Region = kmeans(XGrid, 3, 'MaxIter', 40, 'Start', C);
% ------------------------------

% Plot the results
figure(3);
hold on;
gscatter(score(:,1), score(:,2), idx);
plot(C(:,1), C(:,2), 'kx')
grid on;
title('Clustering Using K-Means', 'Interpreter', 'latex')
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster Centroid', 'Interpreter', 'latex');

figure(4);
% ----- Hierarchical Clustering -----
%create the linkage tree using average-link
Z = linkage(score(:,1:2),'average','euclidean'); 
c = cluster(Z,'maxclust',3);
% -----------------------------------
% See how the cluster assignments correspond to the three species.
crosstab(c, labels);
% Create a dendrogram plot of Z, and visualize it.
dendrogram(Z);
title('Dendrogram Plot of Hierarchical Clustering', 'Interpreter', 'latex')

% Plot the results
figure(5);
hold on;
gscatter(score(:,1), score(:,2), c);
grid on;
title('Clustering Using Hierarchical Clustering', 'Interpreter', 'latex');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Interpreter', 'latex');

figure(6);
% ----------------- GMM clustering -----------------
k = 3; % Number of GMM components
options = statset('MaxIter',1000);

% Options for covariance matrix type
Sigma = {'diagonal','full'}; 
nSigma = numel(Sigma);

% Indicator for identical or nonidentical covariance matrices
SharedCovariance = {true,false}; 
SCtext = {'true','false'};
nSC = numel(SharedCovariance);

d = 500; % Grid length
x1 = linspace(min(score(:,1))-2, max(score(:,1))+2, d);
x2 = linspace(min(score(:,2))-2, max(score(:,2))+2, d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];

threshold = sqrt(chi2inv(0.99,2));
count = 1;
for i = 1:nSigma
    for j = 1:nSC
        gmfit = fitgmdist(score,k,'CovarianceType',Sigma{i}, ...
            'SharedCovariance',SharedCovariance{j},'Options',options); % Fitted GMM
        clusterX = cluster(gmfit,score); % Cluster index 
        mahalDist = mahal(gmfit,X0); % Distance from each grid point to each GMM component
        % Draw ellipsoids over each GMM component and show clustering result.
        subplot(2,2,count);
        h1 = gscatter(score(:,1),score(:,2),clusterX);
        hold on
            for m = 1:k
                idx = mahalDist(:,m)<=threshold;
                Color = h1(m).Color*0.75 - 0.5*(h1(m).Color - 1);
                h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
                uistack(h2,'bottom');
            end    
        plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
        title(sprintf('Sigma is %s\nSharedCovariance = %s',Sigma{i},SCtext{j}),'FontSize',8)
        legend(h1,{'1','2','3'})
        hold off
        count = count + 1;
    end
end
% --------------------------------------------


