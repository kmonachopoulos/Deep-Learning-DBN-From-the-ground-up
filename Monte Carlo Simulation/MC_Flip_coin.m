%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : MC_Flip_coin.m                                         %
%  Description   : Probability coin flip approximation                    %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;
size=1;
pl=1;
N=round(linspace(10,300,4));

for n=1:length(N) % Repeat random generation for N times

% Generate Random variables from Uniform distribution
for i=1:N(n)
    rnd(i)=rand();
end

% Head or tails (zero or one)
rnd=round(size * rnd)+1;

% Initialize histogram
histogram=zeros( 1,size+1);

% Sum the random results to the histogram
for i=1:N(n)   
    histogram(1, rnd(i))=histogram(1, rnd(i))+1;
end

% Normalize and plot the results
subplot(1,4,pl),bar(0:size,(histogram/N(n)));

hold on;
stem(0:size,ones(1,2)*1/2,'r')
hold off

pl=pl+1;

xlabel('x');
ylabel('Probability');
end

suptitle('Flip Coin Distribution Aproximation')

% Compute Statistics
Exp=(0*histogram(1)+1*histogram(1)) / (histogram(1)+histogram(2));
Ph=histogram(1)/sum(sum(histogram));
Pt=(1-Ph);
variance=((0-Exp)^2) * Ph + ((1-Exp)^2) * Ph;

% Print the Results
fprintf('Expectation - Mean  is %.3f\n',Exp) ;
fprintf('Probability of heads is %.3f\n',Ph) ;
fprintf('Probability of tails is %.3f\n',Pt) ;
fprintf('Variance  is %.3f\n',variance);
fprintf('SE is %.3f\n',sqrt(variance));