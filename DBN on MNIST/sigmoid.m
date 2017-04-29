%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                   in the field of computer vision                       %
%  File             : sigmoid.m                	                          %
%  Description      : calculate sigmoid function        				  %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function y = sigmoid(x)

y = 1.0 ./ ( 1.0 + exp(-x) );
