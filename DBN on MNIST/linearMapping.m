%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : LinearMapping.m                                     %
%  Description      : Calculate the linear mapping matrix  between the    %
%                     input data and the output data                      %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function M = linearMapping( IN, OUT )
M = pinv(IN) * OUT;

