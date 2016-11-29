%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Metropolis_Hasting.m                                   %
%  Description   : Random Walk Markov Chain Monte Carlo                   %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear all;close all;

% parameters
NSamples = 10000;        % Number of samples
sig = 0.5;              % Standard deviation of Gaussian proposal
x = -1;                 % Start point
X = zeros(NSamples,1);  % Sequence of samples drawn from the Markov chain
acc = [0 0];            % Vector to track the acceptance rate
bias=0.0;
PRPx=-3:0.1:3;

% Plots
figure;
PRP=exp(-PRPx.^2) .* (2 + sin(PRPx*5) + sin(PRPx*2)); % Calculate P(x)   
subplot(2,2,1),plot(-3:0.1:3,PRP,'r');                % Plot P(x)
norm = normpdf(-3:.1:3,-1,sig);                       % Calculate Norm(ì,ó^2)
hold on;
subplot(2,2,1),plot(-3:.1:3,norm,'b');                % Plot Norm(ì,ó^2)
stem(x,norm(norm==max(norm)));
hold off;
title('First Sample - N(ì,ó^2)')
xlabel('Random Variables')
ylabel('P(x)')
subplot(2,2,3),plot(-3:0.1:3,PRP,'r')

% MH routine
lol=0; % For the first 20 iterations keep thee norms
for i = 1:NSamples
    
    xp = normrnd(x,sig); % Generate candidate from Gaussian
    accprob=(exp(-xp.^2) .* (2 + sin(xp*5) + sin(xp*2)))/...
        (exp(-x.^2) .* (2 + sin(x*5) + sin(x*2))) ; % Acceptance rate
    
    % Draw the first 20 iterations
    if (i<20 && lol==1)
        norm = normpdf(-3:.1:3,x,sig);
        hold on;
        subplot(2,2,3),plot(-3:.1:3,norm);
        stem(x,norm(norm==max(norm)));
        hold off;
        lol=0;
        title('Random Samples - N(ì,ó^2)')
        xlabel('Random Variables')
        ylabel('P(x)')
    end
    
    u = rand; % Uniform random number
    
    if u <= accprob % If accepted
        x = xp; % New point is the candidate
        lol=1;
        a = 1; % Note the acceptance
    else % If rejected
        x = x; % New point is the same as the old one
        a = 0; % Note the rejection
    end
       
    acc = acc + [a 1]; % Track accept-reject status
    % [accepted / all]
    X(i) = x; % Store the i-th sample - sequence 
    
end

% Plots
Exp=hist(X,61);
subplot(2,2,2),bar(linspace(-2,2,61),Exp/200);
title('Expected values HM')
xlabel('Random Variables')
ylabel('Frequency')
hold on;
plot(linspace(-3,3,61),PRP,'r')
hold off;
subplot(2,2,4),plot((X'-bias),1:length(X));
axis([-3 3 1 NSamples])
title('Sample Sequence')
xlabel('Random Variables')
ylabel('Iterations')


