%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : v2hall.m                 		                      %
%  Description      : Transform from visible (input) variables to all     %
%                     hidden (output) variables                           %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function Hall = v2hall(dnn, V)

ind1 = numel(dnn.type);
ind0 = ind1-2;
type = dnn.type(ind0:ind1);

% For Bernoulli - Bernoulli kernel
if( isequal(type, 'RBM') )
    Hall = cell(1,1);
    Hall{1} = v2h( dnn, V );
    
% For DBN iput. Bottom - Up process
elseif( isequal(type, 'DBN') )
    nrbm = numel( dnn.rbm );
    Hall = cell(nrbm,1);
    H0 = V;
    for i=1:nrbm
        H1 = v2h( dnn.rbm{i}, H0 );
        H0 = H1;
        Hall{i} = H1;
    end
end

