predict_label_total = [];
test_label_vector_total = [];

% transpose to make a picture in a row
images2 = images';

% cross validation
accuracy_avg = 0;
for i = 1:5
trIdx = cvo.training(i); %% get the index of training samples
teIdx = cvo.test(i); %% get the index of the test samples
training_label_vector = a(trIdx); %% creating the training label
%ground truth
training_instance_matrix = images2(trIdx,:); %% creating the training
%data matrix
test_label_vector = a(teIdx); %% creating the testing label
%ground truth
test_instance_matrix = images2(teIdx,:); %% creating the test data

% SVM trainining
% RBF kernal
model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.04');

% Linear
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 0');

% SVM predicting
[predict_label, accuracy, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model);
accuracy_avg = accuracy_avg + accuracy;

predict_label_total = [predict_label_total; predict_label];
test_label_vector_total = [test_label_vector_total; test_label_vector];
end
% cross validation accuracy
accuracy_avg = accuracy_avg / 5;

% plot ROC curves
result=plot_roc(predict_label_total,test_label_vector_total);  
disp(result);

