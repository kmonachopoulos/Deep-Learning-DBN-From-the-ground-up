%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : MC_Multiple_fnctns.m                                   %
%  Description   : Central Limit Theorem on random sampling               %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;

% Number of iterations
N=500;

% Plot the functions that we want to find the means by law of large numbers
subplot(3,2,1),plot( (0:1/(N-1):1),sqrt(0:1/(N-1):1),'b.')
xlabel('x ->');
ylabel('f(x)=sqrt(x)');
subplot(3,2,3),plot( (0:1/(N-1):1),(0:1/(N-1):1).^2,'r.')
xlabel('x ->');
ylabel('f(x)=x^2');
subplot(3,2,5),plot((0:1/(N-1):1),1./(pi.*(1+(0:1/(N-1):1).^2)),'g.')
xlabel('x ->');
ylabel('f(x)=1/pi*x^2');
cnt=0;

for i=1:N
    
    r1_h=[];
    r2_h=[];
    r3_h=[];
    
    for j=1:100
        
        cnt=cnt+1;
        
        % Calculate each f(x) for random x
        r1=sqrt(rand());
        r2val=rand();
        r2=r2val*r2val;
        r3=1/(pi*(1+rand()^2));
        
        r1_h(j)=r1;
        r2_h(j)=r2;
        r3_h(j)=r3;
        
        r1_h_v(cnt)=r1;
        r2_h_v(cnt)=r2;
        r3_h_v(cnt)=r3;
    end
    
    means_r1(i)=mean(r1_h);
    means_r2(i)=mean(r2_h);
    means_r3(i)=mean(r3_h);
end

% plot the results
subplot(3,2,2),hist(means_r1,50);
xlabel('N[ì,ó^2]');
ylabel('Sample Means');

subplot(3,2,4),hist(means_r2,50);
xlabel('N[ì,ó^2]');
ylabel('Sample Means');

subplot(3,2,6),hist(means_r3,50);
xlabel('N[ì,ó^2]');
ylabel('Sample Means');

suptitle('Central Limit Theorem in random equations');

fprintf('eq1 -> mean = %.3f eq2 -> mean = %.3f eq3 -> mean = %.3f\n',...
    mean(r1_h_v),mean(r2_h_v),mean(r3_h_v));
fprintf('eq1 -> var = %.3f eq2 -> var = %.3f eq3 -> var = %.3f\n',...
    var(r1_h_v),var(r2_h_v),var(r3_h_v));
fprintf('eq1 -> SE = %.3f eq2 -> SE = %.3f eq3 -> SE = %.3f\n',...
    std(r1_h_v),std(r2_h_v),std(r3_h_v));