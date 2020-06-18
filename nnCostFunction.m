function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels, X, y, lambda)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Variable y in matrics: recode the labels as vectors containing only values 0 or 1,
y_mat = zeros(num_labels, m); 
for (i = 1:m)
  y_mat(y(i),i) = 1;
end

% Feedforward propagation
X1 = [ones(m,1) X];

h2 = sigmoid(Theta1 * X1'); % Output of hidden layer, a size(Theta1, 1) x m matrix
h2 = [ones(m,1) h2'];
h = sigmoid(Theta2 * h2');

% unregularzied cost function
J = (1/m) * sum(sum((-y_mat) .* log(h)-(1-y_mat) .* log(1-h)));

% Regularization term
term1 = sum(sum(Theta1(:,2:end).^2)); % exclude bias term -> 1st col
term2 = sum(sum(Theta2(:,2:end).^2)); % exclude bias term -> 1st col
Regular = (lambda/(2 * m)) * (term1 + term2);

% regularized logistic regression
J = J + Regular;

% 2.3 Backpropagation
Theta1_d = zeros(hidden_layer_size,1);
Theta2_d = zeros(num_labels,1);

for t = 1:m
    % Feedforward propagation
    %disp(size(X));
    a1 = [1; X(t,:)'];
    %disp(size(a1));
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    
    a2 = [1;a2]; % add bias
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % backpropagation
    
    % For each output unit k in layer 3 (the output layer), we set
    delta_3 = a3 - y_mat(:,t);
    
    new = Theta2' * delta_3;
    
    delta_2 = new(2:end) .* sigmoidGradient(z2);
  
    Theta1_d = Theta1_d + delta_2 * a1';
	Theta2_d = Theta2_d + delta_3 * a2';   	
end

% Theta1_grad = Theta1_d / m;
% Theta2_grad = Theta2_d / m;

% Regularization gradient function
reg_term1 = (lambda/m) * [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta1_grad = (Theta1_d / m) + reg_term1;

reg_term2 = (lambda/m) * [zeros(num_labels,1) Theta2(:,2:end)];
Theta2_grad = (Theta2_d / m) + reg_term2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
