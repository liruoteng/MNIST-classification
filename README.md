# PCA Classification
### PCA classification on MNIST dataset
###  README.MD 
####  Author : Ruoteng Li
####  Matric : A0055830J

0. Installation and Dataset
In order to run this script, please make sure MNIST dataset is available at ```mnist``` folder
You may want to go [MNIST Database](http://yann.lecun.com/exdb/mnist/) to download: 

1. Run Project Scripts 
	1 Principal Component Analysis (PCA) - Run Command below (in project root directory)
		>> PCA
	2. Linear Discriminative Analysis (LDA) - Run Command below(in project root directory)
		>> LDA
	3. Support Vector Machine (SVM) - Run Command below (in project root directory)
		>> SVM
	4. Convolutional Neural Networks (CNN) - Run Command below ( in project root directory)
		>> cnn_mnist
		
2. Description of items in the zip folder
	- CNN_output: output results of different architecture. The row number in file name refers to the 	report Table 6. 
	- liblinear-2.1 : linear SVM package
	- libSVM-3.21   : Non linear SVM package
	- mnist         : original MNIST data set
	- output        : CNN output results
	- utils         : private functions used in PCA and LDA
	- practical-cnn--2015a : CNN package