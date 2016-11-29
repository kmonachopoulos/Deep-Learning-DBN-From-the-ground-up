%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Boltzman_Machine_Algorithm.m                           %
%  Description   : BOLTZMANN MACHINE basic algorithm    		          %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%
% ========================================================================
% BOLTZMANN - bias 120 - bipolar values - 120 vissible and 60 hidden units
% ========================================================================

clc; clear; close all;

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

% Concatenate all numbers - Patterns
nums(1,:)=num1;
nums(2,:)=num2;
nums(3,:)=num3;
nums(4,:)=num4;

SizeOfVn=size(nums,2);                  % Number of vissible neurons
SizeOfHn=60;                            % Number of Hidden neurons
SizeOfAlln=SizeOfVn+SizeOfHn;           % Number of All neurons

clamped_v_signals=zeros(1,SizeOfVn);    % Initialize clamped vissible units
unclamped_h_signals=zeros(1,SizeOfHn);  % Initialize unclamped hidden units

normalizeFactor = 0.002;
max_iter= 10;
max_configs=10;
Templow = 0.90;                         % Coefficient for lowering the temperature

% Set rand function given the date
rand('state',sum(100*clock));

% Randomly initialize the weight matrix
W=(2 * rand(1,SizeOfAlln) - 1);
W=(W'*W).*(1-eye(SizeOfAlln));
W=horzcat(W,rand(SizeOfAlln,1)); % concatinate bias vissible - hidden

%%%%%  Weight Matrix %%%%%%%%%
%   raw 1   -> V1 to all | b %
%   raw 2   -> V2 to all | i %
%           .            | a %
%           .            | s %
%           .            |   %
%   raw i   -> Hi to all | t %
%   raw i+1 -> Hi to all | e %
%           .            | r %
%           .            | m %
%           .            | s %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

G=inf; % Kullback distance parameter

while (G>10e-9) % Repeat while error (G) is bigger than 10e-9
    
    for patterns = 1:size(nums,1) % repeat for "patterns" training vectors
        
        % Clumbed vissible units = fixed
        clamped_v_signals=nums(patterns,:);
        
        for confs=1:max_configs % repeat for all configurations
            
            % Set rand function given the date
            rand('state',sum(100*clock));
            
            % Unclumbed hidden units = free
            unclamped_h_signals = (2*round(rand(1, SizeOfHn)) - 1);
            
            % Concatenate vissible and hidden units
            Vc_Hu_signals=[clamped_v_signals unclamped_h_signals];
            
            % =================== POSSITIVE PHASE ===================== %
            T = 10;  % Init temperature
            while (T > 1) % Until no thermal equilibrium
                
                % Initialize Random sampler for the possitive phase
                Rand_idx=find(round(rand(1,SizeOfHn))==1);
                
                for iter=1:max_iter % Reapeat max_iter times
                    
                    % Vissible (fixed) and Hidden units (free) a.k.a
                    % difference of energies - Lyapunov function
                    % Upgrade hidden units at RANDOM. mathematical function [6.2]
                    for i=1:length(Rand_idx)
                        SumUnit=0;
                        for j=1:SizeOfAlln
                            SumUnit= SumUnit+(Vc_Hu_signals(SizeOfVn+Rand_idx(i))...
                                *W(SizeOfVn+Rand_idx(i),j) + W(SizeOfVn+Rand_idx(i),end));
                        end
                        UFixed(i)=SumUnit;
                    end
                    
                    % p(H==1) Hidden unclumped probabillity. mathematical function [6.3]
                    Vc_Hu_Prop = 1./(1 + exp(-UFixed./T));
                    
                    % While T is dicreasing sigmoid function gets tighter
                    % Update neuron status given a probability.
                    for i=1:length(Rand_idx)
                        % Accept the new states with a comparison of the normal
                        % distribution rand() and the sigmoid function
                        if ( Vc_Hu_Prop(i) > rand() )
                            unclamped_h_signals(Rand_idx(i)) = 1;   % accept 1
                        else unclamped_h_signals(Rand_idx(i)) = -1; % accept -1
                        end
                    end
                    % Concatenate vissible and hidden units for the next
                    % iteration.
                    Vc_Hu_signals=[clamped_v_signals unclamped_h_signals];
                    
                end
                T = T*Templow; % decrease the temperature
                fprintf('Possitive phase. Pattern %d . T=%3f\n',patterns,T);     
            end
            
            % Calculate  probabilities propotion to the energies using Gibbs
            % Sampler.
            P_HgivenV(patterns,confs)=exp(CalcEnrgy( Vc_Hu_signals,Vc_Hu_signals',W(1:SizeOfAlln,1:SizeOfAlln)...
                ,W(1:SizeOfAlln,end)));   
        end        
        Fixed_States(patterns,:)=Vc_Hu_signals;     
    end
    
    % =================== END OF POSSITIVE PHASE ============= %
    
    % Calculate distribution over the Data . mathematical function [6.6]
    norm=sum(sum(P_HgivenV));         % Normalization factor - denominator
    P_HgivenV_norm=P_HgivenV./norm;   % Normalize the distribution
    P_Plus=sum(P_HgivenV_norm,2);     % Probability p+(v)
    
    % =================== NEGATIVE PHASE ===================== %
    
    for patterns = 1:size(nums,1) % repeat for "patterns" training vectors
        
        for confs=1:max_configs % repeat for all configurations
            
            % Initialize all units at random
            Vu_Hu_signals=(2*round(rand(1, SizeOfAlln)) - 1);
            
            T = 10;  % Init temperature   
            while (T > 1) % Until no thermal equilibrium
                
                % Initialize Random sampler for the negative phase
                Rand_idxh=find(round(rand(1,SizeOfHn))==1);
                Rand_idxv=find(round(rand(1,SizeOfVn))==1);
                Rand_idx=[Rand_idxv SizeOfVn+Rand_idxh];
                
                for iter=1:max_iter % Reapeat max_iter times
                    
                    % Vissible (free) and Hidden units (free) a.k.a
                    % difference of energies - Lyapunov function
                    % Upgrade hidden units at RANDOM. mathematical function [6.2]
                    for i=1:length(Rand_idxh)
                        SumUnit=0;
                        for j=1:SizeOfAlln
                            SumUnit= SumUnit+(Vu_Hu_signals(SizeOfVn+Rand_idxh(i))...
                                *W(SizeOfVn+Rand_idxh(i),j)+W(SizeOfVn+Rand_idxh(i),end));
                        end
                        UFree_h(i)=SumUnit;
                    end
                    
                    % Vissible (free) and Hidden units (free) a.k.a
                    % difference of energies - Lyapunov function
                    % Upgrade vissible units at RANDOM. mathematical function [6.2]
                    for i=1:length(Rand_idxv)
                        SumUnit=0;
                        for j=1:SizeOfAlln
                            SumUnit= SumUnit+(Vu_Hu_signals(Rand_idxv(i))...
                                *W(Rand_idxv(i),j)+W(Rand_idxv(i),end));
                        end
                        UFree_v(i)=SumUnit;
                    end
                    
                    % Concatinate the units
                    UFree=[UFree_v UFree_h];
                    
                    % p(H==1,V==1) Vissible - Hidden unclumped probabillity. mathematical function [6.3]
                    Vu_Hu_Prop = 1./(1 + exp(-UFree./T));
                    
                    % While T is dicreasing sigmoid function gets tighter
                    % Update neuron status given a probability.
                    for i=1:length(Rand_idx)
                        
                        % Accept the new states with a comparison of the normal
                        % distribution rand() and the sigmoid function
                        if (Vu_Hu_Prop(i) > rand() )
                            Vu_Hu_signals(Rand_idx(i)) = 1;   % accept 1
                        else Vu_Hu_signals(Rand_idx(i)) = -1; % accept -1
                        end
                        
                    end
                end
                T = T*Templow; % decrease the temperature
                fprintf('Negative phase. Pattern %d . T=%3f\n',patterns,T);
            end
            
            % Calculate  probabilities propotion to the energies using Gibbs
            % Sampler.
            P_HandV(patterns,confs)=exp(CalcEnrgy( Vu_Hu_signals,Vu_Hu_signals',W(1:SizeOfAlln,1:SizeOfAlln)...
                ,W(1:SizeOfAlln,end)));
        end               
        Free_States(patterns,:)=Vu_Hu_signals;        
    end    
    
    % =================== END OF NEGATIVE PHASE ============= %

    % Calculate distribution over the Data . mathematical function [6.7]
    norm=sum(sum(P_HandV));         % Normalization factor - denominator
    P_HandV_norm=P_HandV./norm;     % Normalize the distribution
    P_Minus=sum(P_HandV_norm,2);    % Probability p-(v)
    
    % Possitive phase correlation (clamped - unclamped)
    Fixed_States_p = Fixed_States' * Fixed_States;
    
    % Negative phase correlation (unclamped - unclamped)
    Free_States_n = Free_States' * Free_States;
    
    % Difference of correlations is deltaW. This is the partial derivative of
    % the log propabilities of vissible vectors respect to the weights.
    % mathematical function [6.8 - 6.9]
    deltaW = Fixed_States_p - Free_States_n;
    
    % Update weights with this simple form
    
    W = (normalizeFactor/T) .* [W(1:SizeOfAlln,1:SizeOfAlln) ...
        * deltaW .* (1-eye(SizeOfAlln)) W(end,1:SizeOfAlln)'];
    
    
    % Calculate Kullback distance between distributions. mathematical function [6.11]
    G = sum(  (P_Plus)  .* log( P_Plus./ P_Minus));
    fprintf('Kullback Distance between distributions %.10f\n',G);
    
    % Plot hidden weights to identify what the network learned
    for n=1:SizeOfHn
        figure(1),subplot(6,10,n),imagesc( imresize(reshape( W(SizeOfVn+n,1:SizeOfVn),[10,12])' ,[28 28] ))
    end
    suptitle('Hidden units weights')
    colormap('gray')
    
    % Plot the probability distribution of Data and Model
    figure(2),subplot(1,2,1),stem(P_Plus);
    title('P(V) - Data');
    figure(2),subplot(1,2,2),stem(P_Minus);
    title('P''(V) - Model');
    suptitle('Data and Model Distributions over Patterns')
    pause(0.001);
    
end
W