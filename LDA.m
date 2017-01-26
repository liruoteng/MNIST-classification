%% EE5907 LDA for Feature Extraction and Classification
% 
% Author : Ruoteng Li (E0013194)
% Description: 
%
% This script completes following 3 tasts
% 1. Apply LDA to reduce the dimension of vectorized hand written image 
%    to 2 and 3, and visualize the projected data in 2d and 3d plots
%    repectively
%
% 2. Apply LDA to reduce the dimensionality of raw image to 2,3,9
%    respectively and classify them using the rule of Nearest Neighbour.
%
% 3. Find and test the maximum dimensionality that data can be projected
%    via LDA
%
% ==================================================================

% add path
clear; clc;
parpool;
addpath(genpath('utils'));

% Prepare data file
train_img_filename = 'mnist/train-images-idx3-ubyte';
train_lbl_filename = 'mnist/train-labels-idx1-ubyte';
test_img_filename = 'mnist/t10k-images-idx3-ubyte';
test_lbl_filename = 'mnist/t10k-labels-idx1-ubyte';

% In this example, due to the long processing time of NN classification 
% the number of training images are set to 10000 and the number of test
% images is set to 2000 by default. You may feel free to modify the input
% data size. It will improve the classify accuracy, however, the
% processing time will also increase w.r.t the size.
[train_image, train_label] = read_data(train_img_filename, train_lbl_filename, 10000, 0);
[test_image,  test_label] =  read_data(test_img_filename, test_lbl_filename, 2000, 0);

% Variable Initialization
counter = 0;
iterator = 0;
ratio = 0;
accuracy = double(zeros(10,1)); % Result accuracy for each category
accuracy_mat = zeros(3,10);     % Result accuracy map
scrsz = get(groot, 'ScreenSize'); % Get screen width and height

% LDA Step 1. Prepare data matrix
X = train_image;
T = test_image;

% Retrieve data
dimension = size(train_image,1);
Sw = zeros(dimension);
Sb = zeros(dimension);   % Could consider sparse here
N = size(train_image, 2); % Train data size
Nt = size(test_image, 2); % Test data size
Mu = mean(train_image, 2);  % Get mean vector of train data

for i = 0:9
    
    % LDA Step 2. Construct Si matrix of each category
    mask = (train_label ==  i);
    x = X(:, mask);
    ni = size(x, 2);
    pi = ni / N;
    mu_i = mean(x, 2);

    Si = (1/ni) * (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';

    % LDA Step 3. Construct Sw within class covariance
    Sw = Sw + Si * pi;

    % LDA Step 4. Construct Sb between class covariance
    Sb = Sb + pi * (mu_i - Mu) * (mu_i - Mu)';
end

% LDA Step 5. Singular Value Decomposition of Sw\Sb
M = pinv(Sw) * Sb;  % Sw maybe singular, use pseudo-inverse
[U, D, V] = svd(M);

% ===========================
%% Task 1: 2D - Visualization
%
disp('Task 1: Visualize projected data to 2D and 3D plots');

% LDA Step 6 Reduce dimension to 2
R = 2; % Dimensionality
G2 = U(:, 1:R);

% LDA Step 7 Reconstruct the train data matrix
Y2 = G2' * X;

% Plot 2d figure
data2d_fig = figure('Name', '2-D Plot');
set(data2d_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);
for number = 0:9
    
    mask = (train_label ==  number);
    a = Y2(1,mask);
    b = Y2(2,mask);
    c = train_label(mask);
    
    % Draw 2D visualization in separate view
    subplot(2, 5, number+1)
    scatter(a',b');
    title(['Number ', num2str(number)]);
end


%% Task 1b: 3D - Visualization
%
% LDA Step 6 Reduce dimension to 3
R = 3; % Dimensionality
G3 = U(:, 1:R);
Y3 = G3' * X;

% Plot 3d figure
data3d_fig = figure('Name', '3-D Plot');
set(data3d_fig, 'Position', [60 60 scrsz(3)-120 scrsz(4) - 140]);
for number = 0:9
    
    mask = (train_label ==  number);
    a = Y3(1, mask);
    b = Y3(2, mask);
    c = Y3(3, mask);
    subplot(2,5,number+1);
    scatter3(a',b',c');
    title(['Number ', num2str(number)]);
end


%% Task 2a: Classification

disp('Task 2: Apply LDA to reduce dimension of raw data to 2,3,9 respectively');

for p = [2, 3, 9]
    
    % LDA Step 7 Reduce dimension to p 
    G = U(:, 1:p);
    
    % LDA Step 8 Reconstruct train data matrix and test data matrix
    Y = G' * X;
    Y_t = G' * T;
    
    % LDA Step 9 Classify test data using Nearest Neighbor
    accuracy = classifyNN(Y_t, Y, test_label, train_label);
    
    % Display Messages on the screen 
    message = ['Reduced dimension: ', num2str(p), ', ', ...
        'Classification accuracy: ', num2str(sum(accuracy)*100), '%, '];
    
    disp(message);
    
    % restore classification result in accuracy map
    iterator = iterator + 1;
    accuracy_mat(iterator, :) = accuracy;
end

%% Task 2b: LDA enhancement
%

disp('Task 2: Apply PCA first to reduce complexity of original image. ');
disp('Then, apply LDA to reduce dimension of raw data to 2,3,9 respectively');

for P_pca = [10,20,40,80,120,160,200]
    disp(['PCA Dimensionality reduced to: ', num2str(P_pca)]);
    
    Sw = zeros(P_pca);
    Sb = zeros(P_pca);
    X_bar = mean(X, 2);
    S = (X-repmat(X_bar, [1,N])) * (X - repmat(X_bar,[1,N]))' .*(1/N);
    [U_pca, D_pca, V_pca] = svd(S);
    G_pca = U_pca(:, 1:P_pca);

    X_pca = G_pca' * X;
    Mu = mean(X_pca, 2);

    for i = 0:9

        % LDA Step 2. Construct Si matrix of each category
        mask = (train_label ==  i);
        x = X_pca(:, mask);
        ni = size(x, 2);
        pi = ni / N;
        mu_i = mean(x, 2);

        Si = (1/ni) * (x - repmat(mu_i, [1,ni]))*(x - repmat(mu_i, [1,ni]))';

        % LDA Step 3. Construct Sw within class covariance
        Sw = Sw + Si * pi;

        % LDA Step 4. Construct Sb between class covariance
        Sb = Sb + pi * (mu_i - Mu) * (mu_i - Mu)';
    end

    % LDA Step 5. Singular Value Decomposition of Sw\Sb
    M = pinv(Sw) * Sb;  % Sw maybe singular, use pseudo-inverse
    [U, D, V] = svd(M);

    for p = [2, 3, 9]

        % LDA Step 7 Reduce dimension to p 
        G = U(:, 1:p);

        % LDA Step 8 Reconstruct train data matrix and test data matrix
        Y = G' * X_pca;
        Y_t = G' * G_pca' * T;

        % LDA Step 9 Classify test data using Nearest Neighbor
        accuracy = classifyNN(Y_t, Y, test_label, train_label);

        % Display Messages on the screen 
        message = ['Reduced dimension: ', num2str(p), ', ', ...
            'Classification accuracy: ', num2str(sum(accuracy)*100), '%, '];

        disp(message);

        % restore classification result in accuracy map
        iterator = iterator + 1;
        accuracy_mat(iterator, :) = accuracy;
    end

end


%% Task 3: Find maximum dimensionality via LDA
%
% The total number of category in training data set is 10, thus, the
% largest number of classes LDA can achieve is (10-1) = 9. Therefore, let's
% make a test on dimension 10 in order to prove LDA may fail when the
% number of class is more than original categories provided.
%

diag_vec = diag(D);
disp('Now displaying largest 10 eigenvalues of matrix W');
for value = 1:10
    disp(['Eigenvalue ', num2str(value), ': ', num2str(diag_vec(value))]);
end
disp('The 10th largest eigenvalue of matrix W is almost equal to 0, which means');
disp('the objective function J(w) approaches to 0 on dimension 10');
disp('LDA fails to optimize J(w) on 10th dimension. Therefore, it can only achieve 9 classes');

delete(gcp);
% ======= EOF =========

 
    
    
    
