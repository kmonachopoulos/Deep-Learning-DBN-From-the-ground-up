%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : PCfunction.m                                           %
%  Description   : Random Walk Markov Chain Monte Carlo                   %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%
function y = PCfunction(x)
y = exp(-x.^1) .* (2 + sin(x.*5) + sin(x.*2))...
    .* exp(x.^1) .* (2 + sin(x.*2) + sin(x.*8));
end