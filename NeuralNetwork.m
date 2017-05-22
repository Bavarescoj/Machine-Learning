function [J grad] = NeuralNetwork(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   

%For Machine Learning class
%Implements the neural network cost function for a two layer
%neural network which performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% regularization
regul = 0;
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% First, feedforwarding the neural network to get the cost in the
% variable J. 
%
% Then, backpropagation algorithm to compute the gradients for Theta1 and Theta2.
%
% Last but not least, regularization is implemented with the cost function and gradients.

% to calculate hT(x)
% a1 = m x 401 = 5000 x 401
all_a1 = [ones(m, 1) X];
% a2 = 5000 x 25, as a1 = 5000 x 401 and  Theta1' = 401 x 25 (Theta1 = 25 x 401)
all_a2 = sigmoid(all_a1 * Theta1');
% adding bias unit to a2 so a2 = 5000 x 26
all_a2 = [ones(m, 1) all_a2];
% a3 = 5000 x 10, as a2 = 5000 x 26 and Theta2' = 26 x 10 (Theta2 = 10 x 26)
all_a3 = sigmoid(all_a2 * Theta2');

% transform y = 5000x1, so that each training row has 0s and 1s for each class
% which would transform it into y = 5000 x 10
all_Y = zeros(size(y, 1), num_labels);

% putting 1s on all_Y using values of Y as index
for i = 1:m
  all_Y(i, y(i)) = 1;
end

% for each class calculate cost function
for k = 1:num_labels
  % perform vectorized solution of the cost function for each a3 output and then summing all together
  % where all_Y(:,k) and all_a3(:,k) are 5000x1 column vectors 
  J += (1/m) * (((-all_Y(:,k))' * log(all_a3(:,k))) - ((1 - all_Y(:,k))' * log(1 - all_a3(:,k))));
end

% calculating regularization
% Theta1(:, 2:end) has theta without the first column (bias units column), so it's 25x400
% the (:) at the end of Theta1(:, 2:end)(:) unrolls it so it's a 10000(25x400) x 1 vector
regul = lambda/(2*m) * (sum((Theta1(:, 2:end)(:)).^2) + sum((Theta2(:, 2:end)(:)).^2));

% adding regularization to cost function
J += regul;

% -------------------------------------------------------------

% for each training example
for t = 1:m
  
  % -------------- Forward Propagation ------------------------------------  
  % setting a1 to the t-th training example. Column vector of input_layer_size x 1
  a1 = X(t, :)';
  % adding the bias unit a1 = 401 x 1
  a1 = [1; a1];
  
  % calculating z2 variable, where T1 = Sj * (n + 1), T1, being Sj the number of activations nodes
  % z2 = 25 x 1 column vector, where T1 = 25 x 401, and a1 is a (n + 1 = 401) x 1 vector
  z2 = Theta1 * a1;
  % calculating a2 = 25 x 1
  a2 = sigmoid(z2);
  % adding bias unit, a2 = 26 x 1
  a2 = [1; a2];
  
  % calculating z3 variable
  % z3 = 10 x 1 column vector, where T2 = 10 x 26, and a2 = 26 x 1 vector
  z3 = Theta2 * a2;
  % calculating a3 = hT(x) = 10 x 1
  a3 = sigmoid(z3);
  
  % -------------- Back Propagation ------------------------------------  
  % creating a 10 x 1 output column vector
  yt = zeros(num_labels, 1);
  % setting 1 on the class index the current y value belongs 
  yt(y(t), 1) = 1;
  
  % delta3 = 10 x 1, as both a3 and yt are 10 x 1
  delta3 = a3 - yt;
  
  % delta2 = 26 x 1 column vector, as Theta2' = 26 x 10 and delta3 = 10 x 1
  % [1; sigmoidGradient(z2)] as we have to add the bias unit
  delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
  % removing the bias in delta2
  delta2 = delta2(2:end);
  
  % accumulating the gradient 
  Theta1_grad += delta2 * a1';
  Theta2_grad += delta3 * a2';
  
end

% obtainining the (unregularized) gradient for the neural network cost function 
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

% adding the regularization to the gradients for j >= 1
Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
