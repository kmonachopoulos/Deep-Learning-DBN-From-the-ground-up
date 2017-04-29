%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : CalcRmse.m                                          %
%  Description      : Calculate the rmse between predictions and OUTs     %                             
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function [rmse AveErrNum] = CalcRmse( dbn, IN, OUT )

 out = v2h( dbn, IN );                      % Network output
 err = power( OUT - out, 2 );               % Standar error
 rmse = sqrt( sum(err(:)) / numel(err) );	% Mean Square error

 bout = out > 0.5;
 BOUT = OUT > 0.5;

 err = abs( BOUT - bout );                  % Absolute error
 AveErrNum = mean( sum(err,2) );            % Mean error
end
