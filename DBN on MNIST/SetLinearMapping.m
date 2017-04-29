%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : SetLinearMapping.m                                  %
%  Description      : set the RBM associated to the linear mapping to     %
%                     the last layer                                      %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function dbn = SetLinearMapping( dbn, IN, OUT )

% Number of Layers
nrbm = numel(dbn.rbm);
if( nrbm > 1 )
    Hall = v2hall( dbn, IN );
    dbn.rbm{nrbm}.W = linearMapping( Hall{nrbm-1}, OUT );
    dbn.rbm{nrbm}.b = -0.5 * ones(size(dbn.rbm{nrbm}.b));
else
    dbn.rbm{nrbm}.W = linearMapping( IN, OUT );
    dbn.rbm{nrbm}.b = -0.5 * ones(size(dbn.rbm{nrbm}.b));
end
