%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : randRBM.m                                           %
%  Description      : Get a randomized restricted boltzmann machine model %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function rbm = randRBM( dimV, dimH, type )

% Check Network Nodes Kernel
if( ~exist('type', 'var') || isempty(type) )
	type = 'BBRBM';
end

if( strcmpi( 'GB', type(1:2) ) )
    rbm.type = 'GBRBM';
	% Initialize the weights of each Gaussian - Bernoulli RBM
    rbm.W = randn(dimV, dimH) * 0.1;
	% Initialize Hidden Biases of each Gaussian - Bernoulli RBM
    rbm.b = zeros(1, dimH);
	% Initialize Vissible Biases of each Gaussian - Bernoulli RBM
    rbm.c = zeros(1, dimV);
    rbm.sig = ones(1, dimV);
else
    rbm.type = type;
	% Initialize the weights of each Bernoulli - Bernoulli RBM
    rbm.W = randn(dimV, dimH) * 0.1;
	% Initialize Hidden Biases of each Bernoulli - Bernoulli RBM
    rbm.b = zeros(1, dimH);
	% Initialize Vissible Biases of each Bernoulli - Bernoulli RBM
    rbm.c = zeros(1, dimV);
end

