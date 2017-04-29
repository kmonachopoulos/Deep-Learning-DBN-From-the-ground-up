%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : MC_integration_Pi_Estimation.m                         %
%  Description   : Estimate unit circle area A=pi*r^2                     %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%
clc;clear all;close all;

% Plot a rectangular -1 to 1
subplot(1,2,1),hold on;rectangle('Position',[-1 -1 2 2])

% Plot a circle, center 0,0 radius 1
th = 0:pi/50:2*pi;
xunit = 1 * cos(th) + 0;
yunit = 1 * sin(th) + 0;
subplot(1,2,1),hold on;plot(xunit, yunit,'k');
xlabel('x ->');
ylabel('y->');
title('Unit Circle');

% Initialize percentage (in circle), piEstimate
percentage=0;
piEstimate=0;

% Number of Monte Carlo samples
n=10000; 

for i=1:n

    x = 2*rand()-1; % sample the input random variable x
    y = 2*rand()-1; % sample the input random variable y    
    isInside = (x^2 + y^2 < 1); % is the point inside a unit circle?
    
    if isInside == 1
        percentage=percentage+1;
        piEstimate = 4*(percentage/i); % estimate pi ratio       
        subplot(1,2,1),hold on;plot (x,y,'r.');
        hold off;
    else
        subplot(1,2,1),hold on;
        plot (x,y,'b.');
        hold off ;
    end
    
    % Pause to see the intermediate results
    pause(0.01)
    subplot(1,2,2);hold on;plot(i,piEstimate,'g.')
    xlabel('iterations');
    ylabel('pi estimation');
    title(['Estimation pi # ' num2str(piEstimate)])
    hold off;
end

suptitle('Monte Carlo Unit Circle Area Estimation');
