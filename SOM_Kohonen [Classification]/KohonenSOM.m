%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : KohonenSOM.m                                           %
%  Description   : Self Organizing feature Map - Kohonen                  %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear all;close all;

% Data Size
DataSize=1000;

%Neuron Grid Size
NeuGridSz=10;

% Initial Learning rate value
ao=0.3;

% Initial neuron neighbor size
do=5;

% Max number of epochs
T=300;

% Current epoch
t=0;

% Initialize Learning rate history array
LRhistory=zeros(length(T));

% Initialize Neighbor dimensions history array
Dhistory=zeros(length(T));

% Random data points no form
% Set randmon values to 2d input data in limit [0,1]
% rand(1,1000) sto x1 kai sto x2 (Input 1,Input 2)
Input1=rand(1,1000);
Input2=rand(1,1000);

% Random data points in ellipse form
%  r = sqrt(rand(DataSize,1)); % Start out as if filling a unit circle
%  c = 2*pi*rand(DataSize,1);
%
%  X = 3*sqrt(2/59)*r.*cos(c); % Rescale by major & minor semi-axis lengths
%  Y = 3*sqrt(2)*r.*sin(c);
%
%  Input1 = (0.3*X-0.7*Y)/sqrt(100)+0.5; % Rotate back to original coordinates
%  Input2 = (0.7*X+0.3*Y)/sqrt(100)+0.5;


% Randomly initialize the weights in a 10x10 grid. Typical weight values are
% within limit [0.3,0.7] because input values are within limit [0,1], so we
% want the initial weights to form a neighbor in the center of input values.
for j1=1:NeuGridSz
    for j2=1:NeuGridSz
        w1(j1,j2)=0.30+rand*(0.70-0.30);
        w2(j1,j2)=0.30+rand*(0.70-0.30);
    end
end

%% Plotting Training Data
% Plot initial data points and random neurons
figure(1)
plot(Input1,Input2,'.b');
hold on
plot(w1,w2,'or')
plot(w1,w2,'k','linewidth',2)
plot(w1',w2','k','linewidth',2)
xlabel('x1 (input) , w1 (weight)')  % x-axis label
ylabel('x2 (input) , w2 (weight)')   % y-axis label
hold off
suptitle ('Self–Organizing feature Map Training')
title(['current epoch=' num2str(t)]);

% Update figure window and execute pending callbacks
drawnow

%% Train SOM
% while number of epochs is smaller than maximum number of epochs
% recalculate weights and "winner" neuron neighbor for all data inputs.
while (t<=T)
    
    t=t+1;                  % Increase epoch number
    a=ao*(1-t/T);           % recalculate new Learning Rate (decreasing)
    LRhistory(t)=a;         % Save Learning history in every epoch
    d=round(do*(1-t/T));    % recalculate new neighbor size (decreasing)
    Dhistory(t)=d;          % Save neighbor dimensions in every epoch
    
    %loop for the 1000 inputs
    for i=1:DataSize
        
        % Calculate euclidean norm of input (i,j) with all weights
        eucl_norm=(Input1(i)-w1).^2+(Input2(i)-w2).^2; 		% mathematical function [4.1]
        minj1=1;
        minj2=1;
        min_norm=eucl_norm(minj1,minj2); % random initial minimum norm
        
        for j1=1:NeuGridSz
            for j2=1:NeuGridSz
                if eucl_norm(j1,j2)<min_norm % mathematical function [4.2]
                    
                    % Save minimum norm
                    min_norm=eucl_norm(j1,j2);
                    
                    % Find coordinates of minimum norm
                    minj1=j1;
                    minj2=j2;
                end
            end
        end
        
        % Save coordinates of minimum norm
        j1winner= minj1;
        j2winner= minj2;
        
        % Find and Save minimum euclidean norm value and coordinates (the winner neuron)
        % [j1winner,j2winner,min_norm]=find(eucl_norm==min(eucl_norm(:)));
        
        % Update the winning neuron weight using mathematical function [4.4]
        w1(j1winner,j2winner)=w1(j1winner,j2winner)+a*(Input1(i)- w1(j1winner,j2winner));
        w2(j1winner,j2winner)=w2(j1winner,j2winner)+a*(Input2(i)- w2(j1winner,j2winner));
        
        % Update the neighbour neurons using mathematical function [4.6]
        for CurNDist=1:1:d
            
            % Index coordinates of neighbour neurons to the left of the
            % winner neuron
            NeighbourJ1=j1winner-CurNDist;		% mathematical function [4.6]
            NeighbourJ2=j2winner;				% mathematical function [4.6]
            
            % Update the neighbour neurons weights using mathematical function [4.4]
            if (NeighbourJ1>=1)
                w1(NeighbourJ1,NeighbourJ2)=w1(NeighbourJ1,NeighbourJ2)+...
                    a*(Input1(i)-w1(NeighbourJ1,NeighbourJ2));
                w2(NeighbourJ1,NeighbourJ2)=w2(NeighbourJ1,NeighbourJ2)+...
                    a*(Input2(i)-w2(NeighbourJ1,NeighbourJ2));
            end
            
            % Index coordinates of neighbour neurons to the right of the
            % winner neuron
            NeighbourJ1=j1winner+CurNDist;		% mathematical function [4.6]
            NeighbourJ2=j2winner;				% mathematical function [4.6]
            
            % Update the neighbour neurons weights using mathematical function [4.4]
            if (NeighbourJ1<=NeuGridSz)
                w1(NeighbourJ1,NeighbourJ2)=w1(NeighbourJ1,NeighbourJ2)+...
                    a*(Input1(i)-w1(NeighbourJ1,NeighbourJ2));
                w2(NeighbourJ1,NeighbourJ2)=w2(NeighbourJ1,NeighbourJ2)+...
                    a*(Input2(i)-w2(NeighbourJ1,NeighbourJ2));
            end
            
            % Index coordinates of neighbour neurons below the winner neuron
            NeighbourJ1=j1winner;				% mathematical function [4.6]
            NeighbourJ2=j2winner-CurNDist;		% mathematical function [4.6]
            
            % Update the neighbour neurons weights using mathematical function [4.4]
            if (NeighbourJ2>=1)
                w1(NeighbourJ1,NeighbourJ2)=w1(NeighbourJ1,NeighbourJ2)+...
                    a*(Input1(i)-w1(NeighbourJ1,NeighbourJ2));
                w2(NeighbourJ1,NeighbourJ2)=w2(NeighbourJ1,NeighbourJ2)+...
                    a*(Input2(i)-w2(NeighbourJ1,NeighbourJ2));
            end
            
            % Index coordinates of neighbour neurons over the winner neuron
            NeighbourJ1=j1winner;				% mathematical function [4.6]
            NeighbourJ2=j2winner+CurNDist;		% mathematical function [4.6]
            
            % Update the neighbour neurons weights using mathematical function [4.4]
            if (NeighbourJ2<=NeuGridSz)
                w1(NeighbourJ1,NeighbourJ2)=w1(NeighbourJ1,NeighbourJ2)+...
                    a*(Input1(i)-w1(NeighbourJ1,NeighbourJ2));
                w2(NeighbourJ1,NeighbourJ2)=w2(NeighbourJ1,NeighbourJ2)+...
                    a*(Input2(i)-w2(NeighbourJ1,NeighbourJ2));
            end
        end
    end
    
    % For every iteration plot initial data points and neurons using
    % updated weights
    figure(1)
    plot(Input1,Input2,'.b')
    hold on
    plot(w1,w2,'or')
    plot(w1,w2,'k','linewidth',2)
    plot(w1',w2','k','linewidth',2)
    xlabel('Input1 , w1 (weight)')   % x-axis label
    ylabel('Input2 , w2 (weight)')   % y-axis label
    hold off
    suptitle ('Self–Organizing feature Map Training')
    title(['current epoch=' num2str(t)]);
    drawnow
    
end

% Plot Learning rate history
figure(2),plot(1:length(LRhistory),LRhistory);
title('Learning Rate history');
xlabel('Epochs')                        % x-axis label
ylabel('Learning Rate value')           % y-axis label

% Plot Neighbor Dimensions history
figure(3),plot(1:length(Dhistory),Dhistory);
title('Neighbor Dimensions history');
xlabel('Epochs')                        % x-axis label
ylabel('Neighbor Dimensions value')     % y-axis label

%% Clustering Process
fprintf('Clustering Process .... \n');

% Create Random Colour map that refers to each cluster
colourmap=rand(10,10,3);

for i=1:DataSize
    % For each data find the neuron that is closest to the data
    eucl_norm=(Input1(i)-w1).^2+(Input2(i)-w2).^2;
    [j1winner,j2winner,min_norm]=find(eucl_norm==min(eucl_norm(:)));
    
    hold on
    figure(4);
    p=plot(Input1(i),Input2(i),'.b' , 'markersize', 10);
    % set the cluster colour to the input
    set(p,'Color',colourmap(j1winner,j2winner,:));
    % Plot the cluster centers (neuron weights)
    plot(w1,w2,'or')
    xlabel('x1 (input) , w1 (weight)')  % x-axis label
    ylabel('x2 (input) , w2 (weight)')   % y-axis label
    hold off
    pause(0.1)
    suptitle ('Self–Organizing feature Map Training')
    title('Colour Clustering');
end