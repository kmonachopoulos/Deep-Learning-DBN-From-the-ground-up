%-------------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques         %
%                  in the field of computer vision                              %
%  File          : computeCost.m                                                %
%  Description   :                                                              %
%  COMPUTECOST Compute cost for linear regression,                              %
%  J = COMPUTECOST(X, y, weight1,weight2) computes the cost of using theta as   % 
%  the parameter for linear regression to fit the data points in X and y        %
%  Author        : Monachopoulos Konstantinos                                    %
%-------------------------------------------------------------------------------%

function J = computeCost(X, y, weight1,weight2)
m = length(y); % number of training examples

% Compute the cost of a particular choice of weights, we should set J to the cost.
h=weight1.*X(:,2)+ weight2.*X(:,1);
J=(1/2*m)*sum(( h-y ).^2);
end
