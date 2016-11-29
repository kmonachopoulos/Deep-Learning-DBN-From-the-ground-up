%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Perceptron_Linear_Regression.m                         %
%  Description   : Linear Perceptron using Batch Gradient Descent         %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

 clc;clear;close all;

%% Plotting Training Data
fprintf('Plotting Data ...\n')

% Load training set
x = load('ex2x.dat'); 
y = load('ex2y.dat');

% normalization (optional)
%x= (x - ones(length(x),1).*mean(x))./ (ones(length(x),1).* std(x));
%y= (y - ones(length(y),1).*mean(y))./ (ones(length(y),1).* std(y));

x=[ones(length(x),1) x]; % put the bias term

% Plot Data
plot(x(:,2),y,'rx');
xlabel('X -> Input')  % x-axis label
ylabel('Y -> Output') % y-axis label

%% Initialize Linear regression parameters

m = length(y); % number of training examples

% initialize fitting parameters - all zeros
weights=zeros(1,2); % gradient - offset

% Some gradient descent settings
iterations = 1000; % number of iterations
Learning_step_a = 0.07; % step parameter

%% Gradient descent

fprintf('Running Gradient Descent ...\n')

% Compute Gradient descent
[weights(1),weights(2),J_history]=gradientDescent(x, y, weights(1),weights(2), Learning_step_a, iterations);

% Print theta
fprintf('Weights found by gradient descent: %f %f\n',weights(1), weights(2));
fprintf('Minimum of objective function is %f \n',J_history(iterations));

% Vector form w=((xT*x)^-1)*xT)*y
w=(inv(transpose(x)*x)*transpose(x))*y;

% Plot the linear fit
hold on; % keep previous plot visible

% Plot line that produced by minimizing the error
plot(x(:,2), weights(1)*x(:,2)+ weights(2), 'b-'); % scalable variables
plot(x(:,2), w(2)*x(:,2)+w(1), 'k-'); % Matrix variables

% Validate with polyfit fnc that Matlab supports
poly_weights = polyfit(x(:,2),y,1);
plot(x(:,2), poly_weights(1)*x(:,2) + poly_weights(2), 'y--');

title('Linear Regression');
legend('Training data', 'Linear regression', 'Linear regression Matrix','Linear regression with polyfit')
hold off 

%% Prediction 

%Predict values for population sizes of 35,000 and 70,000

predict1 = [1, 3.5] * [weights(1) ; weights(2)];
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * [weights(1) ; weights(2)];
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%% Visualizing J(weights)

fprintf('Visualizing J(weights) ...\n')

% Grid over which we will calculate J
weight1_vals = linspace(-10, 10, 100);
weight2_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(weight1_vals), length(weight2_vals));

% Fill out J_vals
for i = 1:length(weight1_vals)
    for j = 1:length(weight2_vals)
      
	  J_vals(i,j) = computeCost( x, y, weight1_vals(i), weight2_vals(j));
      
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(weight1_vals, weight2_vals, J_vals)
xlabel('weight1'); 
ylabel('weight2');
title('Objective Function Visualization - weight map')

% Contour plot
figure;
[c,h]=contour3(weight1_vals, weight2_vals, J_vals,10);
colorbar;
xlabel('weight1'); ylabel('weight2');
hold on;
title('Objective Function Visualization - weight map minimum')
plot(weights(1), weights(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
