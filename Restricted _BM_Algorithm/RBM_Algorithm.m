%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : RBM_Algorithm.m                                        %
%  Description   : RESTRICTED BOLTZMANN MACHINE basic algorithm           %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%
% ========================================================================
% RBM - bias 120 - bipolar values - 120 vissible and 12 hidden units
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

% Concatenate all numbers
nums(1,:,1)=num1;
nums(2,:,1)=num2;
nums(3,:,1)=num3;
nums(4,:,1)=num4;
nums(nums==-1)=0;


maxepoch=1000;  % number of maximum iterations
numhid=12;      % number of hidden neurons
CD=1;           % Contrastive Divergence steps
err_history=[]; % initialize error history

% Set rand and randn function given the date
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% Learning rate for weights,biases of visible units and hidden units
L_rate = 0.05;

% Normalizations Factor
Norm_fact = 0.002;

% Number of cases (4) and dimensions of each pattern
[numcases, numdims]=size(nums);

% Weights of vissible to hidden units
Weights_v_h = randn(numdims, numhid);

% Biases of hidden units
H_biases = zeros(1,numhid);

% Biases of vissible units
V_biases = zeros(1,numdims);

% Possitive phase hidden units update probabillities
P_h_probs = zeros(numcases,numhid);

% Negative phase hidden units update probabillities
N_h_probs = zeros(numcases,numhid);

% Negative phase vissible units update probabillities
N_v_probs = zeros(numcases,numdims);

% Increasing factor of vissible to hidden units weights
V_h_inc = zeros(numdims,numhid);

% Increasing factor of hidden units biases
H_b_inc = zeros(1,numhid);

% Increasing factor of vissible units biases
V_b_inc = zeros(1,numdims);

for epoch = 1:maxepoch
    
    % Concatinate biases for each vissible unit [patterns x total vissible units]
    V_b = repmat(V_biases,numcases,1);
    % Concatinate biases for each hidden unit [patterns x total hidden units]
    H_b = repmat(H_biases,numcases,1);
    
    % =================== POSSITIVE PHASE ===================== %
    % Possitive phase - activation probabillity of the hidden units P(h|v)
    P_h_probs = 1./(1 + exp(-nums*Weights_v_h - H_b)); % Mathematical function [6.13]
    % Possitive phase - update hidden units states given hidden units probabillities
    P_h_states = P_h_probs > rand(numcases,numhid);
    % =================== END OF POSSITIVE PHASE ============= %
    
    % =================== NEGATIVE PHASE ===================== %
    for CD_counter=1:CD % Using CD (Contrastive Divergence) - 1 step
        % Negative phase - activation probabillity of the vissible units P(v|h)
        N_v_probs = 1./(1 + exp(-P_h_states*Weights_v_h' - V_b)); % Mathematical function [6.14]
        % Negative phase - update vissible units states given vissible units probabillities
        Rec = N_v_probs > rand(numcases,numdims);
        % Negative phase - activation probabillities of the hidden units P(h|Rec)
        N_h_probs = 1./(1 + exp(-Rec * Weights_v_h - H_b));
    end
    
    % Possitive phase statistics - <vihj>0
    P_stats = nums' * P_h_probs;
    % Negative phase statistics - <vihj>1
    N_stats  = Rec' * N_h_probs;
    % =================== END OF NEGATIVE PHASE ============= %
    
    % Extract reconstruction error
    err= sum(sum( (nums-Rec).^2 ));
    err_history(epoch)=err;
    
    % =================== UPDATE WEIGHTS AND BIASES ============= %
    % Vissible - Hidden weights update % Mathematical function [6.17]
    V_h_inc = V_h_inc + L_rate*((P_stats-N_stats)/numcases - Norm_fact * Weights_v_h);
    Weights_v_h = Weights_v_h + V_h_inc;
    % Vissible biases update
    V_b_inc = V_b_inc + (L_rate/numcases)*(sum(nums)-sum(Rec));
    V_biases = V_biases + V_b_inc;
    % Hidden biases update
    H_b_inc = H_b_inc + (L_rate/numcases)*(sum(P_h_probs)-sum(N_h_probs));
    H_biases = H_biases + H_b_inc;
    % =================== END OF UPDATES ============= %
    
    fprintf(1, 'Epoch #%d Reconstruction error #%3.1f\n', epoch, err);
    
    % Plot vissible - hidden weights
    figure(1);
    for i=1:size(Weights_v_h,2)
        subplot(4,4,i),imagesc(imresize(reshape(Weights_v_h(:,i),[10,12])',[28 28]));
    end
    colormap('gray')
    % Plot reconstructions
    figure(1);
    for j=1:size(Rec,1)
        subplot(4,4,i+j),imshow(reshape(Rec(j,:),[10,12])');
    end
    suptitle('Feature Detectors & Reconstructions');
    
    pause(0.01);
    if (err==0)
        
        % Plot the probability distribution of Data and Model
        figure(2)
        subplot(1,2,1),stem(sum(P_h_probs,2)'./(sum(sum(P_h_probs))));
        title('P(V) - Data');
        subplot(1,2,2),stem(sum(N_h_probs,2)'./(sum(sum(N_h_probs))));
        title('P''(V) - Model');
        
        % Plot Reconstruction Error
        figure(3);
        plot(1:epoch,err_history);
        title('Error history');
        xlabel('epochs ->');
        ylabel('Error');
  
        break;
    end
end

