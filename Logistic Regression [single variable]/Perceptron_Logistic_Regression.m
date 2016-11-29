%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Perceptron_Logistic_Regression.m                       %
%  Description   : Logistic Perceptron for binary classification          %
%                  using Batch Gradient Descent                           %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear all;close all;

% Logistic regression is working as linear regression but as an output
% specifies the propability to be attached to one category or the other.
% At the beginning we created a well defined data set that can be easily
% be fitted by a sigmoid function.

%% Create Training Data
% x is the continues input and y is the category of every output [1 or 0]
x = (1:100)';                               % independent variables x(s)
Class_Probability=50;                       % Set the Class Probability
y(1:Class_Probability)  = 0;                % Dependent variables y(s) -- class 0
y(Class_Probability+1:100) = 1;             % Dependent variables y(s) -- class 1
y=y';
y = y(randperm(length(y)));                 % Random order of y array
x=[ones(length(x),1) x];                    % Set the bias term

%% Initialize Logistic regression parameters
m = length(y); % number of training examples

% initialize weights - all zeros
weights = zeros(1,2);

% iterations must be a big number because we are taking very small steps .
iterations = 100000;

% Learning step must be small because the line must fit the data between
% [0 and 1]
Learning_step_a = 0.0005;  % step parameter

%% Gradient descent
fprintf('Running Gradient Descent ...\n')
for iter = 1:iterations
    
    % Logistic hypothesis function
    h = 1 ./ (1 + exp(-(weights(1).*x(:,2)+ weights(2).*x(:,1))));
    
    % Update coefficients
    weights(1)=weights(1) + Learning_step_a * (1/m)* sum((y-h).* x(:,2));
    weights(2)=weights(2) + Learning_step_a * (1/m) *  sum((y-h).*x(:,1));
    
end

% Make a prediction for p(y==1|x==10)
xp=round(100*rand());
fprintf('\nPrediction x = %d , Class = %d \n',xp,round(h(xp)));

%% Validate with matlab function mnrfit
mn_weights=mnrfit(x(:,2),y+1);           % Find Logistic Regression Coefficients
mnrvalPDF = mnrval(mn_weights,x(:,2));   % Define Classes from the Coefficients

% Make a prediction .. p(y==1|x==10)
fprintf('Validate Prediction x = %d , Class = %d \n',xp,round(mnrvalPDF(xp,2)));

%% Visualizing Results
% Plot Logistic Regression Results ...
figure;
subplot(1,2,1),plot(x(:,2),y,'r*');
hold on
subplot(1,2,1),plot(x(:,2),h,'b--');
hold off
title('My Logistic Regression PDF')
xlabel('X -> Continues input');
ylabel('Y -> Propability Density Function');

% Plot Logistic Regression Results (mnrfit) ...
subplot(1,2,2),plot(x(:,2),y,'r*');
hold on
subplot(1,2,2),plot(x(:,2),mnrvalPDF(:,2),'--b')
hold off
title('Validate mnrval Logistic Regression PDF')
xlabel('X -> Continues input');
ylabel('Y -> Propability Density Function');
