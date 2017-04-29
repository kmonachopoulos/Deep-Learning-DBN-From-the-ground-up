%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : StackingRBMs.m                                      %
%  Description      : Get a randomized Deep Belief Nets (DBN) model       %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function dbn = StackingRBMs( dims, type )

% Check Network Nodes Kernel
if( ~exist('type', 'var') || isempty(type) )
	type = 'BBDBN';
end

if( strcmpi( 'GB', type(1:2) ) ) % Gaussian - Bernoulli
 dbn.type = 'GBDBN';             % Gaussian - Bernoulli
 rbmtype = 'GBRBM';              % Gaussian - Bernoulli
else
 dbn.type = 'BBDBN';             % Bernoulli - Bernoulli
 rbmtype = 'BBRBM';              % Bernoulli - Bernoulli
end

% Initialize the RBMs of the DBN as empty cells
dbn.rbm = cell( numel(dims)-1, 1 );

i = 1;
% Initialize the first rbm..
dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );

% Initialize the rest of the RBMs except tht last one..
for i=2:numel(dbn.rbm) - 1
 dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );
end

% Initialize the last RBM..
i = numel(dbn.rbm);
dbn.rbm{i} = randRBM( dims(i), dims(i+1), rbmtype );

end
