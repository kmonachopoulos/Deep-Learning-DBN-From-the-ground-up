%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Gibbs_Sampling.m                                       %
%  Description   : Gibbs Markov Chain Monte Carlo Sampler                 %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%
clc;clear all;close all;

% Sampled Random variables
RandomVariable=[];

% linear coordinates in meshgrid [-256,256]. 2d-sampling
min=-3;max=3;
[xg,yg] = meshgrid(min:0.1:max,min:0.1:max);

% Covariance coefficient between norms
rho=pi/4;

% Change the eccentricity of the normal distribution
th1 = cos(rho)*(xg)-sin(rho)*(yg);
th2 = sin(rho)*(yg)+cos(rho)*(xg);

% Define sigma N[ì,sigma] for each diretion
s1=0.6;
s2=2.2;
m1=0;
m2=0;

% Generate a custom 2D gaussian
Z = exp(-((th1)/s1).^2).*(exp(-((th2)/s2).^2));

% Plot the 2-d normal distribution
contour(xg,yg,Z)

x(1)=2*max*rand()-max; % Initial theta values x1
x(2)=2*max*rand()-max; % Initial theta values x2

% Plot initial point
hold on
plot(x(1),x(2),'s','MarkerFaceColor',[.3 .4 .5],...
    'MarkerEdgeColor',[.3 .4 .5], 'MarkerSize',5 );
title('Markov Chain Monte Carlo Gibbs Sampling');
xlabel('x1');
ylabel('x2');
legend('Mixture of Gaussian','Initial Point');
hold off

for t=1:1000 % Number of samples
    
    old =[x(1) x(2)];                                       % Keep the old values
    x(1) = normrnd( m1+rho*(x(2)-m2) , sqrt(1-rho^2) );     % x1 ~ P( x1(i)|x2(t-1) )
    line([old(1),x(1)],[old(2),x(2)],'Color',[.3 .4 .5]);
    
    old =[x(1) x(2)];                                       % Keep the old values
    x(2)= normrnd(m2+rho*(x(1)-m1),sqrt(1-rho^2));          % x2 ~ P( x2(t)|x1(t) )
    line([old(1),x(1)],[old(2),x(2)],'Color',[.3 .4 .5]);
    
    % Plot the Random variables
    hold on
    plot(x(1),x(2),'r.','markersize',10)
    title('Markov Chain Monte Carlo Gibbs Sampling');
    xlabel('x1');
    ylabel('x2');
    hold off
    
    RandomVariable = [RandomVariable; [x(1) x(2)]]; % Keep sampling variables
    pause(0.001)
end
legend('Mixture of Gaussian','Initial Point','Transitions');
