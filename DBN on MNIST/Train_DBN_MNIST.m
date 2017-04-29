%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : Train_DBN_MNIST.m                                   %
%  Description      : Training DBN with MNIST database                    %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

clear all;clc;close all;

% Check if Database excist for reading
mnistfiles = 0;
if( exist('t10k-images-idx3-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( exist('t10k-labels-idx1-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( exist('train-images-idx3-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( exist('train-labels-idx1-ubyte', 'file') == 2 )
    mnistfiles = mnistfiles + 1;
end
if( mnistfiles < 4 )
    warning('Can not find mnist data files. Please download four data files from http://yann.lecun.com/exdb/mnist/ . Unzip and copy them to same folder as testMNIST.m');
    return;
end

% Read MNIST database. TrainLabels & TestImages & TestLabels
[TrainImages TrainLabels TestImages TestLabels] = mnistread();

% if you want to test with small number of samples, 
% please uncomment the following

TrainNum = 100; % up to 60000
TestNum = 100; % up to 10000
TrainImages = TrainImages(1:TrainNum,:);
TrainLabels = TrainLabels(1:TrainNum,:);
TestImages = TestImages(1:TestNum,:);
TestLabels = TestLabels(1:TestNum,:);

% Set nodes in every Layer of DBN
nodes = [784 800 800 10];

% Create a Random Stacked DBN from multiple RBMs
fprintf('Stacking RBMs....\n');
bbdbn = StackingRBMs( nodes, 'BBDBN' );

% Get the network layers  size
nrbm = numel(bbdbn.rbm);

% Initialize option parameters of the network
Configurations.MaxIter = 100;              % max iterations
Configurations.BatchSize = 10;           % To avoid overfitting 
Configurations.ComPromOut = true;         % Training information output
Configurations.NormFact = 0.0002;         % Normalization Factor
Configurations.HLayer = nrbm-1;           % Store Network hidden layers size 
Configurations.InitialMomentum = 0.5;     % momentum for first five iterations
Configurations.FinalMomentum = 0.9;       % momentum for remaining iterations
Configurations.InitialMomentumIter = 5;

fprintf('Runing Greedy Layer Wise Unsupervised Pre- Training ...\n');
bbdbn = GLWUP(bbdbn, TrainImages, Configurations);

% Set linear Mapping
bbdbn= SetLinearMapping(bbdbn, TrainImages, TrainLabels);

% Run Supervised Fine tuning to Fine - tune the Weights using partial
% patterns target
Configurations.Layer = 0;
fprintf('Runing Supervised Fine tuning using backpropagation...\n');
bbdbn = FineTuning(bbdbn, TrainImages, TrainLabels, Configurations);
fprintf('Done training...\n');

%save('mnistbbdbn.mat', 'bbdbn' );