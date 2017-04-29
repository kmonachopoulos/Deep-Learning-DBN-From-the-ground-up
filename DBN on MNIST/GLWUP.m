%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : GLWUP.m                                             %
%  Description      : Pre-training the Deep Belief Nets (DBN) model by    %
%                     Contrastive Divergence Learning using Greedy Layer  %                     
%                     Wise Unsupervised Pre Training                      %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function dbn = GLWUP(dbn, X, Configurations)

for i=1:Configurations.HLayer+1

	% Pre-train each RBM.
    dbn.rbm{i} = pretrainRBM(dbn.rbm{i}, X, Configurations);
    X0 = X;
    % Take the output of the specific layer to be the input of the next
    % layer.
    X = v2h( dbn.rbm{i}, X0 );
end
