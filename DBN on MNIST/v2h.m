%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : v2h.m                                               %
%  Description      : Transform from visible (input) variables to         %
%                     hidden (output) variables                           %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function H = v2h(dnn, V)

ind1 = numel(dnn.type);
ind0 = ind1-2;
type = dnn.type(ind0:ind1);

% For Bernoulli - Bernoulli kernel
if( isequal(dnn.type, 'BBRBM') )
    % Possitive phase - activation probabillity of the hidden units P(h|v)
    H = sigmoid( bsxfun(@plus, V * dnn.W, dnn.b ) ); % Mathematical function [6.13]
    
% For Gaussian - Bernoulli kernel
elseif( isequal(dnn.type, 'GBRBM') )
    v = bsxfun(@rdivide, V, dnn.sig );
    H = sigmoid( bsxfun(@plus, v * dnn.W, dnn.b ) );
    
% For DBN iput. Bottom - Up process
elseif( isequal(type, 'DBN') )
    nrbm = numel( dnn.rbm );
    H0 = V;
    for i=1:nrbm
        H1 = v2h( dnn.rbm{i}, H0 );
        H0 = H1;
    end
    H = H1;   
end
