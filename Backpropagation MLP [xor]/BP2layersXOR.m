%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : BP2layersXOR.m                                         %
%  Description   : Backpropagation XOR 2 layer perceptorn                 %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear;close all;

% XOR input coordinates for x1 and x2
input = [0 0; 0 1; 1 0; 1 1];

% Desired output of XOR - target variables
target = [0;1;1;0];

% Initialize the bias
bias = [-1 -1 -1];

% Learning coefficient
coeff = 0.7;

% Number of learning iterations (optional)
iterations = 10000;

% In sample mean square error (optional)
MSE = 0.001;

% Calculate weights randomly using seed.
rand('state',sum(100*clock));
weights = -1 +2.*rand(3,3);

% Initialize epoch counter
i=0;

% Initial In sample error variable
Ein=inf;

while (Ein > MSE) && (i<iterations) % While In sample error is bigger than mean square error
    
    i=i+1;
    out = zeros(4,1);
    numIn = length (input(:,1));
    
    for j = 1:numIn % Pass all the features from the network
        
        % Hidden layer
        H1 = bias(1,1)*weights(1,1) ...
            + input(j,1)*weights(1,2)...
            + input(j,2)*weights(1,3);
        
        % Send data through sigmoid function 1/1+e^-x
        % Note that sigma is a different m file
        x2(1) = sigma(H1);
        H2 = bias(1,2)*weights(2,1)...
            + input(j,1)*weights(2,2)...
            + input(j,2)*weights(2,3);
        x2(2) = sigma(H2);
        
        % Target layer
        x3_1 = bias(1,3)*weights(3,1)...
            + x2(1)*weights(3,2)...
            + x2(2)*weights(3,3);
        output(j) = sigma(x3_1);
        
        % Adjust delta values of weights backpropagating the error
        % For the target layer:
        % (1) d(Sj)/d(wij) = xi = output
        % (2) d(e(w))/d(Sj) = d(output-target)/d(Sj) = d(È(Sj)-target)/d(Sj) =
        % = (1-actual target)*(desired target - actual target)= delta.
        % d(Sj)/d(wij)*d(e(w))/d(Sj) = output * delta (chain rule)
        delta3_1 = output(j)*(1-output(j))*(target(j)-output(j));
        
        % Propagate the delta backwards into hidden layers
        delta2_1 = x2(1)*(1-x2(1))*weights(3,2)*delta3_1;
        delta2_2 = x2(2)*(1-x2(2))*weights(3,3)*delta3_1;
        
        % Add weight changes to original weights
        % And use the new weights to repeat process.
        % delta weight = coeff*x*delta
        for k = 1:3
            if k == 1 % Bias cases
                weights(1,k) = weights(1,k) + coeff*bias(1,1)*delta2_1;
                weights(2,k) = weights(2,k) + coeff*bias(1,2)*delta2_2;
                weights(3,k) = weights(3,k) + coeff*bias(1,3)*delta3_1;
            else % When k=2 or 3 input cases to neurons
                weights(1,k) = weights(1,k) + coeff*input(j,1)*delta2_1;
                weights(2,k) = weights(2,k) + coeff*input(j,2)*delta2_2;
                weights(3,k) = weights(3,k) + coeff*x2(k-1)*delta3_1;
            end
        end
        
        %% IN SAMPLE MEAN SQUARE ERROR
        % For every sample - feauture calculate the total error
        Ein=0;
        for erm = 1:numIn
            
            % Hidden layer
            H1 = bias(1,1)*weights(1,1) ...
                + input(erm,1)*weights(1,2)...
                + input(erm,2)*weights(1,3);
            
            % Send data through sigmoid function 1/1+e^-x
            % Note that sigma is a different m file
            x2(1) = sigma(H1);
            H2 = bias(1,2)*weights(2,1)...
                + input(erm,1)*weights(2,2)...
                + input(erm,2)*weights(2,3);
            x2(2) = sigma(H2);
            
            % Target layer
            x3_1 = bias(1,3)*weights(3,1)...
                + x2(1)*weights(3,2)...
                + x2(2)*weights(3,3);
            output(erm) = sigma(x3_1);
            
            % Calculate In - Sample - Error
            Ein = Ein + (output(erm)-target(erm))^2 ;
            
        end
        
        ISE_H(i)=Ein; % Store In - Sample - Error history
    end
end

figure,plot(ISE_H);
title('In sample error history');
xlabel('Iterations')    % x-axis label
ylabel('Error')         % y-axis label
disp(['Mean-Square-Error= ' num2str(Ein)]);
fprintf('\n');

%% Feed forward the network to plot the results
figure;
hold on
for ix=-1:0.1:2;
    for iy=-1:0.1:2;
        
        % Hidden layer
        H1 = bias(1,1)*weights(1,1) ...
            + ix*weights(1,2)...
            + iy*weights(1,3);
        
        % Send data through sigmoid function 1/1+e^-x
        % Note that sigma is a different m file
        % that I created to run this operation
        x2(1) = sigma(H1);
        H2 = bias(1,2)*weights(2,1)...
            + ix*weights(2,2)...
            + iy*weights(2,3);
        x2(2) = sigma(H2);
        
        % target layer
        x3_1 = bias(1,3)*weights(3,1)...
            + x2(1)*weights(3,2)...
            + x2(2)*weights(3,3);
        output = sigma(x3_1);
        
        if(output>0.5)
            plot(ix,iy,'y.');
        else
            plot(ix,iy,'c.');
        end
    end
end

plot(input(target==1,1),input(target==1,2),'r*');
plot(input(target==0,1),input(target==0,2),'b*');
title(' Classification through BackPropagation ')
hold off