%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Perceptron_Polynomial_Regression.m                     %
%  Description   : Linear Perceptron for polynomial Regression            %
%                  using Batch Gradient Descent                           %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear all;close all;

%% Plotting Training Data
% Load training set
fprintf('Plotting Data ...\n')

% Set the dimensions of the plot
dim=5;

% Initialize the independent and the target variable (x,y)
data=[-4.32027649769585 -4.25655976676385;...
    -3.92857142857143 -4.13994169096210;...
    -3.39861751152074 -3.93586005830904;...
    -2.73041474654378 -3.70262390670554;...
    -2.08525345622120 -3.41107871720117;...
    -1.67050691244240 -3.06122448979592;...
    -1.20967741935484 -2.59475218658892;...
    -0.864055299539171 -2.21574344023324;...
    -0.495391705069125 -1.54518950437318;...
    -0.218894009216590 -0.962099125364433;...
    -0.0576036866359448 -0.437317784256561;...
    0.264976958525345 0.466472303206996;...
    0.541474654377881 1.22448979591837;...
    1.00230414746544 2.24489795918367;...
    1.32488479262673 2.76967930029154;...
    1.50921658986175 3.29446064139942;...
    1.78571428571429 3.73177842565597;...
    3.19124423963133 4.37317784256560;...
    3.19124423963133 4.37317784256560];

% Set data in variables
y = data(:, 2);
m = length(y);
X = data(:, 1);

% Plot Data
plot(X, y, 'b+');
axis([-dim dim -dim dim]);

%% Initialize Linear regression parameters
m = length(y); % number of training examples

% Set non linearity to input variable
X = [ones(m, 1), data(:,1), data(:,1).^2, data(:,1).^3 ];

% Some gradient descent settings
iterations = 1000; % number of iterations
Learning_step_a = 0.001; % step parameter

% Extra variables tovfind the optimum fit
J_minimum_error=inf; % minimum error ever recorded through iterations of random weights
Optimum_weights=zeros(4,1); % optimum weights of the minimum error
J_optimum_history=[]; % optimum error history of the minimum error
Optimum_iteration=0; % Random weights iterations

for RandomWeights =1:20
    
    % Initialize Cost function history
    J_history = zeros(1,iterations);
    
    % Initialize weights in random values
    weights = 5*rand(4, 1);
    
    %% Gradient descent
    for iter = 1:iterations
        
        % In every iteration calculate objective function through linear
        % perceptron.
        h=weights(1,1).*X(:,1)+ weights(2,1).*X(:,2)+ ...
            weights(3,1).*X(:,3)+ weights(4,1).*X(:,4);
        
        % Compute Gradient descent and Update weights in every iteration
        temp0 = weights(1,1) - (Learning_step_a/m)*sum((X*weights-y).*X(:,1));
        temp1 = weights(2,1) - (Learning_step_a/m)*sum((X*weights-y).*X(:,2));
        temp2 = weights(3,1) - (Learning_step_a/m)*sum((X*weights-y).*X(:,3));
        temp3 = weights(4,1) - (Learning_step_a/m)*sum((X*weights-y).*X(:,4));
        weights(1,1) = temp0;
        weights(2,1) = temp1;
        weights(3,1) = temp2;
        weights(4,1) = temp3;
        
        % Save Cost function history in every iteration
        J_history(iter) =(1/(2*m)) * sum(  (h - y).^2);
    end
    
    % Print weight values
    fprintf('Weights found by gradient descent: %f %f %f %f\n',weights(1), weights(2),weights(3), weights(4));
    fprintf('Minimum of objective function is %f \n',J_history(iterations));
    
    % If smaller error is recorded using the specific weights
    if J_minimum_error > J_history(iterations)
        
        J_minimum_error=J_history(iterations); % Store the minimum error
        J_optimum_history=J_history; % Store the history of the minimum error
        Optimum_weights=weights; % Store the optimum weights
        Optimum_iteration=RandomWeights; % Store optimum iteration number
    end
    
    % Plot the non linear function on the initial data
    hold on
    plot(-dim:0.01:dim, weights(1) + (-dim:0.01:dim).*weights(2) + (-dim:0.01:dim).^2.*weights(3) + (-dim:0.01:dim).^3.*weights(4), 'y--');
    hold off
end

% Plot the minimum error ever recorded
fprintf('\nMinimum error of all iterations %f \n',J_minimum_error);

% Plot the optimum non linear function on the initial data
hold on
title('Polynomian regression');
xlabel('X -> Input')  % x-axis label
ylabel('Y -> Output') % y-axis label
plot(-dim:0.01:dim, Optimum_weights(1) + (-dim:0.01:dim).*Optimum_weights(2)...
    + (-dim:0.01:dim).^2.*Optimum_weights(3) + ...
    (-dim:0.01:dim).^3.*Optimum_weights(4), 'r--');
hold off

legend('Training data', 'Polynomian regression - random initial points');

%% Visualizing J(weights)
fprintf('Visualizing J(weights) ...\n')

% Visualize 2D error history and final error
figure, plot(1:iter,J_optimum_history);
hold on;
plot(iter,J_optimum_history(end),'ro');
hold off;
xlabel('X -> Iterations')  % x-axis label
ylabel('Y -> Error') % y-axis label
title(['Error history - iterations :' num2str(iter)])
axis([-1 iter+1 J_optimum_history(end)-100 J_optimum_history(1)+100])
str_f = sprintf('final error %0.5f ',J_optimum_history(end));
text(iter,J_optimum_history(end),horzcat(str_f,'\rightarrow  '),'HorizontalAlignment','right');
