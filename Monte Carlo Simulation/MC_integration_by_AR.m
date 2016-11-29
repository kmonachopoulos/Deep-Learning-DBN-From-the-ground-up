%-------------------------------------------------------------------------%
%  Master thesis : Image Processing using NN                              %
%  File          : MC_integration_by_AR.m                                 %
%  Description   : Estimate under the curve area using A/R method         %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;

% Set the function to be estimated
x=0:0.0001:1;
fx=x.^2;

% Plot the function
plot(x,fx)
hold on

% Initiallize the collector - estimator
collect=0;

% Initiallize total iterations
N=10000;

for l=1:N
    % Set a random point inside the limits [0,1]
    x=rand();
    y=rand();
    
    % if the point is under the curve note it as collected (green)
    if (y<x^2)
        collect=collect+1;
        plot(x,y,'g.');
        xlabel('x');
        ylabel('f(x)=x^2');
    else 
        % else do not vote for it and just print (blue)
        plot(x,y,'b.')
    end
    pause(0.001)
end
estimation=collect/N;
suptitle(['Monte Carlo Area Under the Curve Estimation # ' num2str(estimation)])

% print the collected random spots over all random spots, this is the area under
% te curve
fprintf('Area under the curve is %.3f\n',estimation)