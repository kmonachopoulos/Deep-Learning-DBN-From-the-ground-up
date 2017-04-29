%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : MC_Roll_die.m                                          %
%  Description   : Rolling a Die approximation                            %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;

die_out_history=[];

for i = 1:500 % Repeat 500 times
    
    for j = 1:100 % Roll the die 100 times 
        rnd_die_out(j) = 1+round(5*rand());
    end
    
    % Keep the means
     die_avr(i) = mean(rnd_die_out);

    % Save the rolling outputs
    die_out_history = [die_out_history rnd_die_out];  
    
    % Plot Std Error
    hold on
    plot(i,sqrt(var(die_avr))/sqrt(i*j),'b.');
    hold off
    pause(0.01)
    
    title('Standar Error')
    xlabel('Trials ->')
    ylabel('std/sqrt(n)')
end

% Expected values through the experiement
figure;
Exp=hist(die_out_history,6);
subplot(1,2,1),bar(Exp/(i*j));
title('Expected values of probabilities')
xlabel('Die Output')
ylabel('Frequency')
hold on;
stem(1:6,ones(1,6)*1/6,'r')
hold off

% Central Limit Theorem
CLT=hist(die_avr, 6);
subplot(1,2,2),bar(CLT);
title('Central Limit Theorem - Binomial')
xlabel('Die means')
ylabel('Frequency')
suptitle('Rolling a die Distribution Aproximation')

% Print the Results
fprintf('Mean  is %.3f\n',mean(die_avr)) ;
fprintf('Variance  is %.3f\n',var(die_avr));
fprintf('Standar error is %.3f\n',sqrt(var(die_avr))/sqrt(i*j));