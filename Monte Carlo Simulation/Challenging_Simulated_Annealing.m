%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Metropolis_Hasting.m                                   %
%  Description   : Random Walk MCMC - Simulated Annealing Optimization    %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;

% parameters
NSamples = 100;    % Number of samples
sig = 0.5;         % Standard deviation of Gaussian proposal
x = 2*rand()-1;    % Start point
X = [];            % Sequence of samples drawn from the Markov chain
counter=0;
T=20;              % Initial temperature
lowerbound=-3;     % Upper bound
uperbound=3;       % Lower bound

while (T>0.01)
    
    for i = 1:NSamples
        
        counter=counter+1;
        
        Energynow=PCfunction(x);
        xp=(6*rand())-3;
%         xp = normrnd(x,sig); % Generate candidate from Gaussian
        
        if (xp<lowerbound)
            xp=lowerbound;
        end
        if (xp>uperbound)
            xp=uperbound;
        end
        
        Energyprop=PCfunction(xp);
        
        u = rand; % Uniform random number
        De=Energyprop-Energynow;
        
        if (De<0)               % always accept negative difference
            x = xp;             % New point is the candidate
        elseif (De>0)
            if(exp(-De/T) > u)  % If accepted
                x = xp;         % New point is the candidate
            else                % If rejected
                x = x;          % New point is the same as the old one
            end
        end
        X(counter) = x;         % Store the i-th sample - sequence
        
    end
    
    % Plots
    PRPx=-3:0.1:3;
    PRP=PCfunction(PRPx);
    plot(-3:0.1:3,PRP,'r');                % Plot P(x)     title('Optimization - Rocks!!')
    title('Markov Chain Monte Carlo Simulated Annealing')
    xlabel('Random Variables')
    ylabel('P(x)')
    hold on;
    stem(x, PCfunction(x) )
%     norm = normpdf(-3:.1:3,x,sig);
%     plot(-3:.1:3,norm,'b');
    hold off;
    pause(0.1)
    T=T*0.99 % decrease temperature
    
end
