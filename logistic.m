function [loss,gradient]=logistic(w,xTr,yTr)
% function w=logistic(w,xTr,yTr)
%
% INPUT:
% xTr dxn matrix (each column is an input vector)
% yTr 1xn matrix (each entry is a label)
% w weight vector (default w=0)
%
% OUTPUTS:
% 
% loss = the total loss obtained with w on xTr and yTr
% gradient = the gradient at w
%

%[d,n]=size(xTr);

e = exp(-yTr.*(w'*xTr));
loss=sum(log(1+e));
gradient=   sum(-yTr.*xTr.* e./(1+e),2); 