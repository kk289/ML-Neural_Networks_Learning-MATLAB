# Machine Learning (MATLAB) - Neural Networks Learning

Machine Learning course from Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning).

### Environment
- macOS Catalina (version 10.15.3)
- MATLAB 2018 b

### Dataset
- ex4data1.mat
- ex4weights.mat

### Files included in this repo
- ex4.m - Octave/MATLAB script that steps you through the exercise 
- ex4data1.mat - Training set of hand-written digits
- ex4weights.mat - Neural network parameters for exercise 4 
- submit.m - Submission script that sends your solutions to our servers 
- displayData.m - Function to help visualize the dataset
- fmincg.m - Function minimization routine (similar to fminunc) 
- sigmoid.m - Sigmoid function
- computeNumericalGradient.m - Numerically compute gradients 
- checkNNGradients.m - Function to help check your gradients 
- debugInitializeWeights.m - Function for initializing weights 
- predict.m - Neural network prediction function

[⋆] sigmoidGradient.m - Compute the gradient of the sigmoid function 

[⋆] randInitializeWeights.m - Randomly initialize weights

[⋆] nnCostFunction.m - Neural network cost function

## Part 1: Neural Networks
In previous part, We implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we provided. In this part, we will implement the *backpropagation algorithm* to learn the parameters for the neural network.

For this portion we will use following MATLAB script
```
ex4.m
```

### Dataset

Given dataset *ex4data1.mat* contains 5000 training examples of handwritten digits, where each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector. 

Each of these training examples becomes a single row in our data matrix X. This gives us a 5000 by 400 matrix X where every row is a training example for a handwritten digit image.

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.

```
% Load saved matrices from file
load('ex4data1.mat');
% The matrices X and y will now be in Octave environment
```

### Part 1.1: Visualizing the data
#### displayData.m - Function to help visualize the dataset

In ex4.m, the code randomly selects selects 100 rows from X and passes those rows to the displayData function. This function maps each row to a 20 pixel by 20 pixel grayscale image and displays the images together. 

![plot](https://github.com/kk289/ML-Multiclass_Classification_and_Neural_Network-MATLAB/blob/master/Figure/datavisualize.jpg)
- Figure: Dataset

### Part 1.2: Model representation
Our neural network has 3 layers – an input layer, a hidden layer and an output layer. Recall that our inputs are pixel values of digit images. Since the images are of size 20×20, this gives us 400 input layer units (excluding the extra bias unit which always outputs +1). As before, the training data will be loaded into the variables X and y.

For this portion we will use following MATLAB script
```
ex4.m
```
```
% Load saved matrices from file
load('ex4weights.mat');
% The matrices Theta1 and Theta2 will now be in workspace
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
```

### Part 1.3: Feedforward and cost function
We will implement the cost function for the neural network with *unregularization*

The cost function for the neural network (without regularization):  
![costfunction](Figure/costfunction.png)

#### nnCostFunction.m - Neural network cost function
Implements the neural network cost function for a two layer neural network which performs classification

```
% Variable y in matrics: recode the labels as vectors containing only values 0 or 1,
y_mat = zeros(num_labels, m); 
for (i = 1:m)
  y_mat(y(i),i) = 1;
end
```

```
% Feedforward propagation
X = [ones(m,1) X];

h2 = sigmoid(Theta1 * X'); % Output of hidden layer, a size(Theta1, 1) x m matrix
h2 = [ones(m,1) h2'];
h = sigmoid(Theta2 * h2');
```

```
% unregularization cost function
J = (1/m) * sum(sum((-y_mat) .* log(h)-(1-y_mat) .* log(1-h)));
```

Result: 
Feedforward Using Neural Network ...
Cost at parameters (loaded from ex4weights): *0.287629* 


### Part 1.4: Regularized cost function 
We will implement the cost function for neural networks with *regularization*

The cost function for the neural network (without regularization):  
![costfunction_reg](Figure/costfunction_reg.png)

```
% regularized cost function

% Regularization term
term1 = sum(sum(Theta1(:,2:end).^2));
term2 = sum(sum(Theta2(:,2:end).^2));
Regular = (lambda/(2 * m)) * (term1 + term2);

% regularized logistic regression
J = J + Regular;
```

Result: 
Checking Cost Function (w/ Regularization) ... 
Cost at parameters (loaded from ex4weights): 0.383770 


## Part 2: Backpropagation
We will implement the backpropagation algorithm to compute the gradient for the neural network cost function.

Let start with implementing the sigmoid gradient function. 
![gradient](Figure/gradient.png)

```
% sigmoid function
g = 1.0 ./ (1.0 + exp(-z));
```

#### sigmoidGradient.m - Compute the gradient of the sigmoid function 
```

```


Once we have computed the gradient, we will be able to train the neural network by minimizing the cost function J(Θ) using an advanced optimizer such as fmincg.

First implement the backpropagation algorithm to compute the gradients for the parameters for the (unregularized) neural network.

```
% unregularized gradient function for neural network


```

```
% regularized gradient function for neural network


```


## Course Links

1) Machine Learning by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning)

2) [Neural Network Learning](https://www.coursera.org/learn/machine-learning/home/week/5) 
(Please notice that you need to log in to see the programming assignment.)# ML-Neural_Networks_Learning-MATLAB