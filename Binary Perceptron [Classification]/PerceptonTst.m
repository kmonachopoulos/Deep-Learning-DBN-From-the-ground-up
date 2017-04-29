%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : PerceptronTst.m                                        %
%  Description   : Test the perceptron with the test data and measure     %
%                  the out of sample error                                %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

function e=PerceptonTst(x,y,w,b)

%% Testing phase
[l,p]=size(x);
e=0; % number of test errors

for i=1:l          
    xx=x(i,:); % Pass each test data xt to the network
    ey=xx*w+b; % Apply the Percepton Transfer Function with the optimum weights
    
    if ey>=0   % Sign each data to a Class
       ey=1;
    else
       ey=-1;
    end   
    if y(i)~=ey; % Set Out-Of-Sample error if data is misclassified
       e=e+1;
    end;
end