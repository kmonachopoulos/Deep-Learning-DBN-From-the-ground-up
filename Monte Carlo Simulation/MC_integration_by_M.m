%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : MC_integration_by_M.m                                  %
%  Description   : Estimate under the curve area using Mean Value method  %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;

% Set the function to be estimated
x=0:0.0001:1;
fx=x.^2;

% Plot the function
subplot(1,2,1),plot(x,fx);
xlabel('x');
ylabel('f(x)=x^2');

% Initiallize total iterations
N=500;
for i=1:N
    
    y=[];
    x=[];
    
    for j=1:50        
        % Set a random point inside the limits [0,1]
        x(j)=rand();
        y(j)=x(j)^2;            
    end
    
    Sample(i)=mean(y);
    hold on;
    subplot(1,2,2),plot(i,mean(Sample),'.g');
    xlabel('Sample iterations ->');
    ylabel('Mean of samples');
    hold off;
    pause(0.1)
end

suptitle(['Monte Carlo Area Under the Curve Estimation M = ' num2str(mean(Sample))])

% print the collected random spots over all random spots, this is the area under
% te curve
fprintf('Area under the curve is %.3f\n',mean(Sample))