%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : PerceptronTrn.m                                        %
%  Description   : Train the perceptron finding the optimum weights with  %
%                  the smallest in sample error or within 1000 iterations %
%  Author        : Monahopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

function [OptW,b,iterations,ISE_H,Weight_H]=PerceptonTrn(x,y)

[l,p]=size(x);
w=rand(p,1);  % initialize weights
b=0;          % initialize bias
ier=1;        % initialize a misclassification indicator
iterations=0; % number of iterations
n=0.3;        % learning rate

while ier==1, %repeat until no internal error occurs
       ier=0; 
       e=0; % number of training errors
       iterations=iterations+1; % Increase iterations in every step 

       for i=1:l  % a pass through x           
           xx=x(i,:); % Pass every observation x to the network
           ey=xx*w+b; % estimate y
           
           % Perceptron threshold. 
           if ey>=0
              ey=1;
           else
              ey=-1;
           end
           
           if y(i)~=ey;                  % miss-classification
              er=y(i)-ey;                % error difference [target - output]
              w=w'+ double(er*n)*x(i,:); % Update weigths
              e=e+1 ;                    % number of training errors
              w=w'; 
           end;
       end; 
       
      %% Find the best weights corresponding to minimum error - "Pocket Algorithm" 
    
       ee=e;    % number of training errors
       if (iterations == 1)
           OptError=ee;
		   OptW=w;
       else
           if(ee<OptError)
               OptW=w;
           end
       end
      
       if ee>0  % Continue if there is still errors
          ier=1;           
       end
       
       % Store In-sample-error and weights
       ISE_H(iterations)=ee;
       Weight_H(:,iterations)=w;
       
       if iterations==10000 % Stop after 10000 iterations if algorithm does not converge
          ier=0;
       end;
 end;
 
 disp(['In-Sample-Error=' num2str(e) '     Training data Size=' num2str(l)])


 