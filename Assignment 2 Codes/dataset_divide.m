clear;
clc;

% load the dataset
load mnist-1-5-8.mat;

% preparing the vectors for digit 5 against the rest
a = zeros(size(labels));
a(labels==5) = 1;

% split the data into 5 fold.
cvo = cvpartition(a,'KFold',5);