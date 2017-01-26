% EE5907 SVM Classification
% 
% Author : Ruoteng Li (E0013194)
% Description: 
%
% This script completes following 3 tasts
% 1. Apply linear SVM to do classification on the lower dimension vectors
%    pre-processed by PCA. The dimension is 40 80 and 200 respectively.
%  
% 2. Apply Kernel SVM with radial basis kernel to do classification on the
%    lower dimension vectors of MNIST data. The dimension is 40 80 and 200.
%    
%    Note: For both method, the cost penalty parameter iterates from 0.01
%    to 10 on logarithmical basis. For Kernel SVM, the gamma value is
%    changed to find out the best performance. 
%
% ==================================================================

%
% add path
clear; clc;
addpath(genpath('utils'));
addpath(genpath('libsvm-3.21'));
addpath(genpath('liblinear-2.1'));

% Prepare data file
train_img_filename = 'mnist/train-images-idx3-ubyte';
train_lbl_filename = 'mnist/train-labels-idx1-ubyte';
test_img_filename = 'mnist/t10k-images-idx3-ubyte';
test_lbl_filename = 'mnist/t10k-labels-idx1-ubyte';

% load data set
[train_image, train_label] = read_data(train_img_filename, train_lbl_filename, 10000, 0);
[test_image, test_label] = read_data(test_img_filename, test_lbl_filename, 2000, 0);

% Variable Initialization
counter = 0;
iterator = 0;
ratio = 0;     % Total energy preservation ratio
distance = 0;  % Euclidean distance for Nearest Neighbor
scrsz = get(groot,'ScreenSize'); % Get screen width and height

% 1. Prepare data matrix
X = train_image;
T = test_image;

% Retrieve dimension and sample number
[d,N] = size(X);
[td, tn] = size(T);

% 2. Create covariance matrix S 
X_bar = mean(X, 2);
S = (X-repmat(X_bar, [1,N])) * (X-repmat(X_bar,[1,N]))' .* (1/N);

% 3. Singular Value Decomposition of S
%    Get Projection matrix U
[U, D, V] = svd(S);


% 4. Use liblinear SVM to do classification

% Run linear SVM
disp('Run liblinear SVM');
for p = [40,80,200]
    
    % Project the original images to lower dimension
    G = U(:, 1:p);
    Y = X' * G;
    Yt = T' * G;
    Y = sparse(Y);
    Yt = sparse(Yt);
    
    for c = [0.01, 0.1, 1, 10]
        
        % Use linear kernel for large data set
        arg1 = strcat({'-s 2 -c '}, num2str(c), ' -q');
        linear_model = train(train_label, Y, arg1{1});
        [linear_label, linear_accuracy, v] = predict(test_label, Yt, linear_model, '-q');

        message1 = strcat('Reduced dimension: ', ...
            num2str(p), ' Penalty Parameter: ', num2str(c), ...
            ' Linear Accuracy: ', num2str(linear_accuracy(1)));
        disp(message1);
    
    end
end


% Run libsvm-3.21 SVM
disp('Run Radial Basis Kernel SVM');
for p = [40,80,200]
    
    % Project the original images to lower dimension
    G = U(:, 1:p);
    Y = X' * G;
    Yt = T' * G;
    Y = sparse(Y);
    Yt = sparse(Yt);
    
    for c = [0.01, 0.1, 1, 10]
        for gamma = [0.01, 0.1, 1]
            % Use non-linear kernel with different Gamma value
            arg2 = strcat({'-t 2 -c '}, num2str(c), {' -g '}, num2str(gamma), {' -h 0 -q'});
            nonlinear_model = svmtrain(train_label, Y, arg2{1});
            [nonlinear_label, nonlinear_accuracy, decision_values] = svmpredict(test_label, Yt, nonlinear_model, '-q');

            message2 = strcat('Reduced dimension: ', num2str(p), ...
                ' Penalty Parameter: ', num2str(c), ...
                ' Gamma Parameter: ', num2str(gamma), ...
                ' Non-linear Accuracy: ', num2str(nonlinear_accuracy(1)));
            disp(message2);
        end
    end
end

