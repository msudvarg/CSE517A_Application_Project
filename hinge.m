function [loss,gradient,preds]=hinge(w,xTr,yTr,lambda)
% function w=ridge(xTr,yTr,lambda)
%
% INPUT:
% xTr dxn matrix (each column is an input vector)
% yTr 1xn matrix (each entry is a label)
% lambda regression constant
% w weight vector (default w=0)
%
% OUTPUTS:
%
% loss = the total loss obtained with w on xTr and yTr
% gradient = the gradient at w
%

%[d,n]=size(xTr);

Val=  max(1-yTr.*(w'*xTr),0);
loss= sum(Val) + lambda*(w'*w); 
 
idx=(Val>0);  
gradient=sum(-1.*yTr(idx).*xTr(:,idx),2) + 2*lambda.*w; 
