% EE5907 PCA for Feature Extraction Visualization and Classification
% 
% Author : Ruoteng Li (E0013194)
% Description: 
%
% This script completes following 3 tasts
% 1. Apply PCA to reduce the dimension of vectorized hand written image 
%    to 2 and 3, and visualize the projected data in 2d and 3d plots
%    repectively
%
% 2. Apply PCA to reduce the dimensionality of raw image to 40,80,200
%    respectively and classify them using the rule of Nearest Neighbour.
%
% 3. Reduce the raw image to dimension d with total energy preservation
%    over 95%. Find the d value and report the classification result based on
%    Nearest Neighbor
%
% ==================================================================

% % add path
clear; clc;
addpath('utils');
parpool('local', 2); 

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
ratio = 0;     % Total energy preservation ratio
distance = 0;  % Euclidean distance for Nearest Neighbor
accuracy_mat = zeros(3,10);      % Result accuracy map
scrsz = get(groot,'ScreenSize'); % Get screen width and height

% 1. Prepare data matrix
X = train_image;
T = test_image;

% Retrieve dimension and sample number
[d,N] = size(X);
[td, tn] = size(T);

% 2. Create covariance matrix S 
X_bar = mean(X, 2);
% Here, there is another approach by only calculate covariance of X
% S = cov(X);
S = (X-repmat(X_bar, [1,N])) * (X-repmat(X_bar,[1,N]))' .* (1/N);

% 3. Singular Value Decomposition of S
%    Get Projection matrix U
[U, D, V] = svd(S);
diag_vec = diag(D);


%% Task 1a: 2D - Visualization
% 
disp('Task 1: Visualize projected data to 2D and 3D plots respectively');

p = 2;
% PCA Step 4. Reduce dimension to 2
G2 = U(:, 1:p);

% 5. Reconstruct train data matrix
Y2 = G2' * X;

% Plot 2d figure
data2d_fig = figure('Name', '2-D Plot');
set(data2d_fig,'Position',[40 60 scrsz(3)-80 scrsz(4)-140]);

% The color is used to draw classes in one graph
% color = [ 0 0 0; 0 0 1; 0 1 0 ; 0 1 1 ; 1 0 0 ; 1 0 1 ; 1 1 0 ; 0.3 0.4 0.6; ...
%     1 0.55 0; 0.5 0.5 0.5];
for number = 0:9
    
    mask = (train_label ==  number);
    a = Y2(1,mask);
    b = Y2(2,mask);
    c = train_label(mask);
    
    % Draw 2D visualization in separate view
    subplot(2,5,number+1);       % Add plot in 2 x 5 grid
    scatter(a', b');
    title(['Number ' , num2str(number)]);
    
% Draw 2D visualization in one graph
%     graph_2d(number+1) = scatter(a', b',[], c, '+');
%     graph_2d(number+1).MarkerEdgeColor = color(number+1, :);
%     hold on;
%     title('PCA 2D Visualization');
end

% the code below this comment is used to add legend to 2D visualization graph
% 
% hold off;
% legend([graph_2d(1),graph_2d(2),graph_2d(3),graph_2d(4),graph_2d(5),...
%     graph_2d(6),graph_2d(7),graph_2d(8),graph_2d(9),graph_2d(10)],...
%     'Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', ...
%     'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9', ...
%     'Location', 'southwest', 'FontSize', 12, 'FontWeight', 'bold');

    
% Plot eigen vectors of sample variance
eigen2d_fig = figure('Name', '2D Eigen Plot');
set(eigen2d_fig,'Position',[40 100 scrsz(3)-80 scrsz(4)-220]);
eig1 = reshape(G2(:,1), [28,28]);
eig2 = reshape(G2(:,2), [28,28]);
subplot(1,2,1);
imagesc(eig1)
title('1st Eigen vectors of Sample Variance');
subplot(1,2,2);
imagesc(eig2);
title('1st Eigen vectors of Sample Variance');


%% Task 1b: 3D - Visualization
%
p = 3;
% 4. Reduce dimension to 3
G3 = U(:, 1:p);

% 5. Reconstruct train data matrix
Y3 = G3' * X;

% Plot 3d figure
data3d_fig = figure('Name','3-D Plot'); 
set(data3d_fig,'Position',[40 60 scrsz(3)-80 scrsz(4)-140]);
for number = 0:9
    mask = (train_label ==  number);
    a = Y3(1,mask);
    b = Y3(2,mask);
    c = Y3(3,mask);
    color = train_label(mask);
    subplot(2,5,number+1);       % add first plot in 2 x 1 grid
    scatter3(a', b', c', [],color, '.');
    title(['Number ' , num2str(number)]);

end

% Plot eigen vectors of sample variance
eigen3d_fig = figure('Name', '3D Eigen Plot');
set(eigen3d_fig,'Position',[40 100 scrsz(3)-80 scrsz(4)-220]);
eig1 = reshape(G3(:,1), [28,28]);
eig2 = reshape(G3(:,2), [28,28]);
eig3 = reshape(G3(:,3), [28,28]);
subplot(1,3,1);
imagesc(eig1);
title('1st Eigen vectors of Sample Variance');
subplot(1,3,2);
imagesc(eig2);
title('2nd Eigen vectors of Sample Variance');
subplot(1,3,3);
imagesc(eig3);
title('3rd Eigen vectors of Sample Variance');
%  

%% Task 2:  Classification

disp('Task 2: Apply PCA to reduce dimension of raw image to 40,80,200 respectively');

% Different dimensionality
for p = [40, 80, 200]
    
    % 4. Reduce dimention to p
    G = U(:, 1:p);

    % 5. Reconstruct train data matrix and test data matrix
    Y = G' * X;
    Y_t = G' * T;
    
    % 6. Classify test data using Nearest Neighbor    
    accuracy = classifyNN(Y_t, Y, test_label, train_label);
    
    % Calculation of Total Energy Preservation
    ratio = sum(diag_vec(1:p, 1)) / trace(D);
    
    % Display Messages on the screen 
    message = ['Reduced dimension: ', num2str(p), ', ', ...
        'Classification accuracy: ', num2str(accuracy*100), '%, ' ...
        'Total energy preservation: ', num2str(ratio), '/1.0'];
    
    disp(message);
    
    % restore classification result in accuracy map
    iterator = iterator + 1;
    accuracy_mat(iterator) = accuracy;
end

%% Task 3: Find Energy Preservation over 95%
% 
disp('Task 3: Find eigen value that preserves 95% total energy and classify');

tr = trace(D);
sz = size(diag_vec, 1);
energy = 0; % current energy
idx = 0;  % index where the eigen value preserves 95% total energy

% Find the eigen value that preserves over 95% total energy
for i = 1:sz
    energy = energy + diag_vec(i,1);
    if energy / tr >= 0.95
        idx = i;
        break;
    end
end 

% idx = 0 if not found, error
if idx == 0
    error('Could not find corresponding eigen value');
end

disp(['Dimension: ', num2str(idx)]);

% PCA Step 4. Reduce dimention to p
Gx = U(:, 1:idx);

% PCA Step 5. Reconstruct train data matrix and test data matrix
Yx = Gx' * X;
Yxt = Gx' * T;

% PCA Step 6. Classify test data using Nearest Neighbor
accuracy = classifyNN(Yxt, Yx, test_label, train_label);

% Display Messages on the screen 
message = ['Reduced dimension: ', num2str(idx), ', ', ...
    'Classification Accuracy: ', num2str(accuracy*100), '%, ' ...
    'Total energy preservation: ', num2str(energy/tr), '/1.0'];

disp(message);

delete(gcp);
% ============= EOF ================

