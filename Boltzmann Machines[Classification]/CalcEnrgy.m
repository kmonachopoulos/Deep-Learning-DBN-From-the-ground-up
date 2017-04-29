%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%  Master thesis : Image Processing using NN                              %
%  File          : CalcEnrgy.m                                            %
%  Description   : Set noise with a propabilistic NoiseRatio coefficient  %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

function [ Energy ] = CalcEnrgy( Si,Sj,Wij,b)

    Energy = (-(1/2)*(Si*Wij*Sj)) - (sum(Si*b));  

end
