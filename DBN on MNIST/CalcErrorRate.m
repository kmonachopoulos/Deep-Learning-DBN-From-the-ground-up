%-------------------------------------------------------------------------%
%  Master thesis    : Research and development on Deep Learning techniques%
%                     in the field of computer vision                     %
%  File             : CalcErrorRate.m                                     %
%  Description      : Calculate error rate                                %
%  Author           : Monachopoulos Konstantinos                           %
%  Original Author  : The algorithm is inspired by Masayuki Tanaka        %
%-------------------------------------------------------------------------%

function ErrorRate = CalcErrorRate( dbn, IN, OUT )

 out = v2h( dbn, IN );
 [m ind] = max(out,[],2);
 out = zeros(size(out));
 
 % Copy output patterns from the database
 for i=1:size(out,1)
  out(i,ind(i))=1;
 end
 
 % Calculate error rate from the difference of outputs
 ErrorRate = abs(OUT-out);
 ErrorRate = mean(sum(ErrorRate,2)/2);

end

