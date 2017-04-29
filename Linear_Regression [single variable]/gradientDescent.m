%-------------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques         %
%                  in the field of computer vision                              %
%  File          : gradientDescent.m                                            %
%  Description   :                                                              %
%  GRADIENTDESCENT Performs gradient descent to learn theta,                    %
%  theta = GRADIENTDESENT(x, y, weight1,weight2, Learning_step_a, num_iters)    %
%  updates theta by taking num_iters gradient steps with learning rate alpha    %
%  Author        : Monachopoulos Konstantinos                                    %
%-------------------------------------------------------------------------------%

function [weight1,weight2, J_history] = gradientDescent(x, y,weight1,weight2,Learning_step_a, num_iters)

% Initialize Objective Function History
J_history = zeros(num_iters, 1);
m = length(y); % number of training examples

    % run gradient descent
    for iter = 1:num_iters

        % In every iteration calculate objective function through linear
        % perceptron (1.2), b->bias=1.
        h=weight1.*x(:,2)+weight2.*x(:,1);

        % Update weights using Gradient Descent update rule. 
        weight1 = weight1 - Learning_step_a * (1/m) *  sum((h-y).* x(:,2)); %(1.11)-(1.12)
        weight2 = weight2 - Learning_step_a * (1/m) *  sum((h-y).* x(:,1)); %(1.11)-(1.12)

        % Save the cost J in every iteration
        J_history(iter) = computeCost(x, y, weight1,weight2);
    end
end
