%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : sigma.m       		                                  %
%  Description   : Non Linear Sigmoid activation function                 %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

function [ y ] = sigma( x )
y=1/(1+exp(-x)) ;
end

