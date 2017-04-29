%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : mnistread.m                             			  %
%  Description      : Readding the MNIST database                 		  %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function [ TrainImages, TrainLabels, TestImages, TestLabels ] = mnistread( mnistfilenames )

% Check if MNIST database exist and fetch the names of the files
if( ~exist('mnistfilenames') )
    mnistfilenames = cell(4,1);
    
    % Name files of MNIST database
    mnistfilenames{1} = 'train-images-idx3-ubyte';
    mnistfilenames{2} = 'train-labels-idx1-ubyte';
    mnistfilenames{3} = 't10k-images-idx3-ubyte';
    mnistfilenames{4} = 't10k-labels-idx1-ubyte';
end

% Store the train images
TrainImages = mnistimageread( mnistfilenames{1} );
% Store the train Labels
TrainLabels = mnistlabelread( mnistfilenames{2} );
% Store the test images
TestImages = mnistimageread( mnistfilenames{3} );
% Store the test Labels
TestLabels = mnistlabelread( mnistfilenames{4} );

% Cast the train images to Single format
TrainImages = single(TrainImages)/255.0;
% Cast the train images to Single format
TestImages = single(TestImages)/255.0;

TrainLabels = single(TrainLabels);
TestLabels = single(TestLabels);

end

% Read the images from MNIST database
function images = mnistimageread( imagefile )
    
    % File manipulation procedures
    fid = fopen( imagefile, 'rb');
    magic = fread(fid, 1, '*int32',0,'b');
    nimgs = fread(fid, 1, '*int32',0,'b');
    nrows = fread(fid, 1, '*int32',0,'b');
    ncols = fread(fid, 1, '*int32',0,'b');
    images = fread(fid, inf, '*uint8',0,'b');
    fclose( fid );
    
    % Check if MNIST database is on the correct format
    if( magic ~= 2051 )
        warning( sprintf( '%s is not MNIST image file.', imagefile ) );
        images = [];
        return;
    end
    
    images = reshape( images, [nrows*ncols, nimgs] )';
    for i=1:nimgs
        % Format images as Strings
        img = reshape( images(i,:), [nrows ncols] )';
        images(i,:) = reshape(img, [1 nrows*ncols]);
    end
end

% Read the Labels of each image
function labels = mnistlabelread( labelfile )
    fid = fopen( labelfile, 'rb');
    magic = fread(fid, 1, '*int32',0,'b');
    nlabels = fread(fid, 1, '*int32',0,'b');
    ind = fread(fid, inf, '*uint8',0,'b');
    fclose( fid );
    
    if( magic ~= 2049 )
        warning( sprintf( '%s is not MNIST label file.', labelfile ) );
        labels = [];
        return;
    end
    
    labels = zeros( nlabels, 10 );
    ind = ind + 1;
    for i=1:nlabels
        labels(i, ind(i)) = 1;
    end
end


