# MNIST Classification
### PCA, LDA, SVM classification on MNIST dataset
##### Ruoteng Li
##### 11.11.2016

#### 0. Installation and Dataset
- In order to run this script, please make sure MNIST dataset is available at ```mnist``` folder. You may want to go [MNIST Database](http://yann.lecun.com/exdb/mnist/) to download: 
- Please install MATLAB 2014b or later version

#### 1. Run Project Scripts 
- Principal Component Analysis (PCA) - Run Command below (in project root directory)
```
>> PCA
```
- Linear Discriminative Analysis (LDA) - Run Command below(in project root directory)
```
>> LDA
```
- Support Vector Machine (SVM) - Run Command below (in project root directory)
```
>> SVM
```
		
#### 2. Description of directories 
	- liblinear-2.1 : linear SVM package
	- libSVM-3.21   : Non linear SVM package
	- mnist         : original MNIST data set
	- utils         : private functions used in PCA and LDA
