function [loss,gradient]=ridge(w,xTr,yTr,lambda)
% function w=ridge(xTr,yTr,lambda)
%
% INPUT:
% w weight vector (default w=0)
% xTr dxn matrix (each column is an input vector)
% yTr 1xn matrix (each entry is a label)
% lambda regression constant
%
% OUTPUTS:
% loss = the total loss obtained with w on xTr and yTr
% gradient = the gradient at w
%
% [d,n]=size(xTr);
% 
% loss=w'*(xTr*xTr')*w-2*w'*xTr*yTr' + yTr*yTr' + lambda.*w'*w; 
% 
% gradient= 2*(xTr*xTr')*w-2*xTr*yTr' + 2*lambda.*w;


[d,n]=size(xTr);

preds = w' * xTr;
loss = sum((preds - yTr) .^ 2) + lambda * (w' * w);
gradient = 2*xTr*(preds-yTr)' + 2*lambda.*w;
