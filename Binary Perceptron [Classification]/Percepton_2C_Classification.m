%-------------------------------------------------------------------------%
%  Master thesis : Research and development on Deep Learning techniques   %
%                  in the field of computer vision                        %
%  File          : Perceptron_2C_Classification.m                         %
%  Description   : Adaline Binary Perceptron for classification           %
%  Author        : Monachopoulos Konstantinos                              %
%-------------------------------------------------------------------------%

clc;clear all;close all;

%% Generate 2 dimensions linear separable data 
%   You may change the size of the data from here or input your own data
%   note that the drawing is for two dimensions only, hence you need to 
%   modify the code for different data.

mydata = rand(600,2);

% Separate the data into two classes
% Every X,Y or Y,X difference point must be greater than 0.012 
% this way we remove the data points that X and Y are very close 
% (diagonal - line).

acceptindex = abs(mydata(:,1)-mydata(:,2))>0.012;
mydata = mydata(acceptindex,:); % Keep the data that is linear seperetable

% label the data to [-1,1] proportion to X > Y (down - right, under the conceivable diagonal)
myclasses = int8(mydata(:,1)>mydata(:,2));      % Set the first class to 1
myclasses(myclasses==0)=-1;                     % Set the second class to -1
[m n]=size(mydata);

% training data
x=mydata(1:400,:);   
y=myclasses(1:400);

% test data
xt=mydata(401:m,:); 
yt=myclasses(401:m);

%% Train the Percepton
[OptW,b,iterations,ISE_H,Weight_H] = PerceptonTrn(x,y);
iterations

%% Test the Percepton
e=PerceptonTst(xt,yt,OptW,b);
disp(['Out-Of-sample-Error=' num2str(e) '     Test Data Size= ' num2str(m-400)])


%% Draw the result (sparating hyperplane)
l=y==-1; % Index l holds zeros points of y flaged as -1.
figure,plot(x(l,1),x(l,2),'kx' );
hold on
plot(x(~l,1),x(~l,2),'ro');
[l,p]=size(x);
axis([0 1 0 1]), axis square, grid on
title(' Linear Classification through Percepton ')
legend('Class A','Class B');
plotpc(OptW',b)
hold off

% Plot the In sample error
figure,plot(1:iterations,ISE_H,'r-');
legend('Ein');
title('In-Sample-Error-History')
xlabel('iterations -> ');
ylabel('In-Sample-Error ');
grid on;

% Plot the weight history
figure,plot(1:iterations,Weight_H,'b-');
legend('Weights');
title('Weight-History');
xlabel('iterations -> ');
ylabel('Weights ');
grid on;
