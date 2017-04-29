%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : Evaluation_DBN_MNIST.m                              %
%  Description      : Evaluating DBN with MNIST database                  %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

clear all;

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

[TrainImages TrainLabels TestImages TestLabels] = mnistread();

load mnistbbdbn;

rmse= CalcRmse(bbdbn, TrainImages, TrainLabels);
ErrorRate= CalcErrorRate(bbdbn, TrainImages, TrainLabels);
fprintf( 'For training data:\n' );
fprintf( 'rmse: %g\n', rmse );
fprintf( 'ErrorRate: %g\n', ErrorRate );

rmse= CalcRmse(bbdbn, TestImages, TestLabels);
ErrorRate= CalcErrorRate(bbdbn, TestImages, TestLabels);
fprintf( 'For test data:\n' );
fprintf( 'rmse: %g\n', rmse );
fprintf( 'ErrorRate: %g\n', ErrorRate );

