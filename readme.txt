Instructions for the codes:

by Zhang Ce, 11910803, EE Department, SUSTech, edited: 2022/1/2

For Question 1 (PCA & Clustering part):

        You can run Assignment2_part1.m directly in MATLAB.


For Question 2 (LDA & Clustering part):

        You can run Assignment2_part2.m directly in MATLAB.


For Larger MNIST dataset PCA & LDA:

        You can run Assignment2_additional1.m(PCA),  Assignment2_additional2.m(LDA) directly in MATLAB.


For Question 3 (Binary Classification):

        Firstly, you need to run dataset_divide.m to divide the dataset, then follow the instructions below.

        3-1 SVM with RBF kernel:

	You can run Assignment2_part3_1.m directly in MATLAB.

        3-2 SVM with linear kernel:

	Change the codes to 

	model = svmtrain(training_label_vector, training_instance_matrix, '-t 0'); 

	in Assignment2_part3_1.m line 22 and re-run in MATLAB, or you can annotate line 22 and use the code in line 25.

        3-3 Neural Network:

	You can run Assignment2_part3_2.m directly in MATLAB.

