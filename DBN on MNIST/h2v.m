%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : h2v.m                                               %
%  Description      : Transform from hidden (output) variables to         %
%                     visible (input) variables                           %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function V = h2v(dnn, H)

ind1 = numel(dnn.type);
ind0 = ind1-2;
type = dnn.type(ind0:ind1);

if( isequal(dnn.type, 'BBRBM') )
    % Negative phase - activation probabillity of the vissible units P(v|h)
    V = sigmoid( bsxfun(@plus, H * dnn.W', dnn.c ) );   % Mathematical function [6.14]
    
% For Gaussian - Bernoulli kernel  
elseif( isequal(dnn.type, 'GBRBM') )
    h = bsxfun(@times, H * dnn.W', dnn.sig);
    V = bsxfun(@plus, h, dnn.c );
    
% For DBN iput. Up - Down process
elseif( isequal(type, 'DBN') )
    nrbm = numel( dnn.rbm );
    V0 = H;
    for i=nrbm:-1:1
        V1 = h2v( dnn.rbm{i}, V0 );
        V0 = V1;
    end
    V = V1;
    
end
