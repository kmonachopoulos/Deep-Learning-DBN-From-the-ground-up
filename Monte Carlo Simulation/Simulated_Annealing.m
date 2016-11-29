%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Simulated_Annealing.m                                  %
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
T=1;               % Initial temperature
lowerbound=-1;     % Upper bound
uperbound=1;       % Lower bound

while (T>0.01)
    
    for i = 1:NSamples
        
        counter=counter+1;       
        Energynow=Pfunction(x);    
        xp = normrnd(x,sig); % Generate candidate from Gaussian
        
        if (xp<lowerbound)
            xp=lowerbound;
        end
        if (xp>uperbound)
            xp=uperbound;
        end
        
        Energyprop=Pfunction(xp);        
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
    PRP=Pfunction(PRPx);
    plot(-3:0.1:3,PRP,'r');                % Plot P(x)     title('Optimization - Rocks!!')
    title('Random Walk Markov Chain Monte Carlo Simulated Annealing')
    xlabel('Random Variables')
    ylabel('P(x)')
    hold on;
    stem(x, Pfunction(x) )
    norm = normpdf(-3:.1:3,x,sig);
    plot(-3:.1:3,norm,'b');
    hold off;
    pause(0.1)
    T=T*0.99 % decrease temperature
    
end

% VALIDATION
matx=simulannealbnd(@Pfunction,0,-1,1);
hold on;
stem(matx, Pfunction(matx));
norm = normpdf(-3:.1:3,matx,sig);
plot(-3:.1:3,norm,'y');
hold off;
