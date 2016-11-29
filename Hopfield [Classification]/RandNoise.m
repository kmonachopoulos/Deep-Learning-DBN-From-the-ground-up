%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : RandNoise.m                                            %
%  Description   : Set noise with a propabilistic NoiseRatio coefficient  %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

function [NoiseChArray] = RandNoise (CharArray, Length, NoiseRatio)

for k=1:1:Length
    if (rand()<NoiseRatio)
        CharArray(k) = -CharArray(k);
    end
end

NoiseChArray=CharArray;