Y_total = [];
test_label_vector_total = [];

% define feedforward network
net = feedforwardnet(2, 'traingd');

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

% Configure the net
net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0; % validation set [%]
net.divideParam.testRatio = 0; % test set [%]
net.inputs{1}.processFcns = {}; % modify the process function for inputs
net.outputs{2}.processFcns = {}; % modify the process function for outputs
net.layers{1}.transferFcn = 'logsig'; % the transfer function for the first layer
net.layers{2}.transferFcn = 'logsig'; % the transfer function for the second layer
net.trainParam.lr = 0.8; % learning rate. You may need to adjust it in the experiment.

% train the network
net = train(net, training_instance_matrix', training_label_vector');

% get the prediction result
Y = (sim(net, test_instance_matrix')).'; % return a vector of outputs

Y_total = [Y_total; Y];
test_label_vector_total = [test_label_vector_total; test_label_vector];

% threshold for output
Y(Y > 0.5) = 1;
Y(Y <= 0.5) = 0;

% get predicted accuracy
num_correct = 0;
for j = 1:120
    if Y(j,1) == test_label_vector(j,1)
        num_correct = num_correct + 1;
    end
end
accuracy = num_correct / 120;
disp(accuracy)
accuracy_avg = accuracy_avg + accuracy;
end
accuracy_avg = accuracy_avg / 5;

% plot the ROC curve
result=plot_roc(Y,test_label_vector);  
disp(result);
grid on;


