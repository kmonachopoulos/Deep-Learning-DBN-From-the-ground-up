%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Hopfield_Bin.m                                         %
%  Description   : Character recognition using hopfield network           %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear all;close all;

%Define a standard vector of 4 characters [1-2-3-4]
% Number 1 array
num1=  [-1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1  1  1  1 -1 -1 -1 -1 ...
    -1 -1  1  1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1  1  1  1  1  1  1 -1 -1];

% Number 2 array
num2=  [-1 -1 -1 -1  1  1  1 -1 -1 -1 ...
    -1 -1  1  1  1  1  1  1 -1 -1 ...
    -1  1  1 -1 -1 -1 -1  1  1 -1 ...
    -1  1 -1 -1 -1 -1 -1 -1  1  1 ...
    1 -1 -1 -1 -1 -1 -1  1  1 -1 ...
    -1 -1 -1 -1 -1 -1  1  1 -1 -1 ...
    -1 -1 -1 -1 -1  1  1 -1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1 -1  1  1 -1 -1 -1 -1 -1 ...
    -1 -1  1  1 -1 -1 -1 -1 -1 -1 ...
    -1  1  1 -1 -1 -1 -1 -1 -1 -1 ...
    1  1  1  1  1  1  1  1  1  1];

% Number 3 array
num3=  [-1  1  1  1  1  1  1  1  1 -1 ...
    -1 -1 -1 -1 -1 -1 -1 -1  1 -1 ...
    -1 -1 -1 -1 -1 -1  1  1 -1 -1 ...
    -1 -1 -1 -1  1  1 -1 -1 -1 -1 ...
    -1 -1  1  1 -1 -1 -1 -1 -1 -1 ...
    1  1  1  1  1  1  1  1 -1 -1 ...
    -1 -1 -1 -1 -1 -1 -1 -1  1 -1 ...
    -1 -1 -1 -1 -1 -1 -1 -1 -1  1 ...
    -1 -1 -1 -1 -1 -1 -1 -1  1 -1 ...
    1 -1 -1 -1 -1 -1 -1 -1  1 -1 ...
    -1  1 -1 -1 -1 -1 -1  1 -1 -1 ...
    -1 -1  1  1  1  1  1 -1 -1 -1];

% Number 4 array
num4=  [-1 -1 -1 -1 -1  1  1  1 -1 -1 ...
    -1 -1 -1 -1  1  1  1  1 -1 -1 ...
    -1 -1 -1  1  1 -1  1  1 -1 -1 ...
    -1 -1 -1  1  1 -1  1  1 -1 -1 ...
    -1 -1  1  1 -1 -1  1  1 -1 -1 ...
    -1 -1  1  1 -1 -1  1  1 -1 -1 ...
    -1  1  1 -1 -1 -1  1  1 -1 -1 ...
    1  1 -1 -1 -1 -1  1  1 -1 -1 ...
    1  1  1  1  1  1  1  1  1  1 ...
    1  1  1  1  1  1  1  1  1  1 ...
    -1 -1 -1 -1 -1 -1  1  1 -1 -1 ...
    -1 -1 -1 -1 -1 -1  1  1 -1 -1];

% Concatenate all numbers
nums(1,:)=num1;
nums(2,:)=num2;
nums(3,:)=num3;
nums(4,:)=num4;

% Initialize Weight matrix
W=zeros(120);

% Hebb Rule weight matrix initialization
for i=1:4
    Xi=nums(i,:);
    Xj=nums(i,:)';
    W=W+(Xj*Xi); % mathematical function [4.8]    
end

% Normalize weight matrix
W=W/120;

% Set diagonal elements to zero so non of the neurons is connected to itself
W=W.*(~eye(120));  % mathematical function [4.9]

% Set the bias term is the mean of the different pattern components
bias=(max(num1)+min(num1))/2;

% Create a noise threshold
n=0.30;

% Energy matrix
Energy=[];
InitEnergy=zeros(size(nums,1),1);

for i=1:4
% Calculate Energy
InitEnergy(i)=CalcEnrgy(nums(i,:),nums(i,:)',W,bias);   % mathematical function [4.11]
end     

% Initialize noise input matrix and result character matrix
NoiseInput=zeros(size(nums));
ResultChar=zeros(size(nums));
NoiseChrUpd=zeros(1,size(nums,2));

% Set random noise to all characters by noise threshold coefficient
for i=1:size(nums,1)
    NoiseInput(i,:)=RandNoise(nums(i,:),size(nums,2),n);
end

% Train hopfield network
for i=1:size(NoiseInput,1)
    
    % Fetch each noise character
    NoiseChr=NoiseInput(i,:);
    
    % Set Updating flag to "true".
    UpdatingFlag=1;
    iterations=0;
    
    % While Updating flag is set to "true" and maximum iterations is less
    % than 10000 -> Keep updating neuron status.
    while (UpdatingFlag==1 && iterations<100)
        
        % Increace iteration value
        iterations=iterations+1;
        
        % Set update flag to zero
        UpdatingFlag=0;
        
        %% Asynchronus Hopfield network training
        % Update each neuron at a time
        for k=1:size(W,2)
                hi=0;
                for j=1:size(W,1)
                    hi=hi+(W(k,j)*NoiseChr(j));
                end
                NoiseChrUpd(k)=sign(hi); % Threshold state [+1,-1]           
        end
       
        		% Calculate Energy
        Energy(i,iterations)=CalcEnrgy(NoiseChr,NoiseChr,W,bias);   % mathematical function [4.11]
        
        if (iterations>1)
            % dE is always negative minimizing the energy untill network
            % reach steady state !!
             
            dE(i,iterations-1)=Energy(i,iterations)-Energy(i,iterations-1);
        end
        
        % Check if the new neurons status is the same as the previous
        % status. If not update the neurons status else training is
        % finished for that pattern.
        if ~isequal(NoiseChrUpd,NoiseChr)
            UpdatingFlag=1;
            NoiseChr=NoiseChrUpd;
        end        
    end  
    ResultChar(i,:)=NoiseChr;
end

%% Plot the results
figure(1);
origcoor=[1 4 7 10];
for i=1:length(origcoor)

    subplot(4,3,origcoor(i)),imshow(reshape(nums(i,:),[10,12])')
    title('Original character');
end

corcooor=[2 5 8 11];
for i=1:length(corcooor)
    subplot(4,3,corcooor(i)),imshow(reshape(NoiseInput(i,:),[10,12])')
    title('Noise Character');
end

resultcoor=[3 6 9 12];
for i=1:length(resultcoor)
    ResultChar( ResultChar==-1 )=0;
    subplot(4,3,resultcoor(i)),imshow(reshape(ResultChar(i,:),[10,12])')
    title('Noise Character after network converge');
end
suptitle('Character Recognition - Hopfield Net')

