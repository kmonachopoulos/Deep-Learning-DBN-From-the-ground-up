%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Pfunction.m                                            %
%  Description   : Random Walk Markov Chain Monte Carlo                   %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%
function y = Pfunction(x)
y = exp(-x.^2) .* (2 + sin(x.*5) + sin(x.*2));
end