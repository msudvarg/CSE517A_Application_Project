function traintwitter(xTr,yTr);
%function trainspamfilter(xTr,yTr);
% INPUT:	
% xTr
% yTr: 1 is Trump, -1 is Obama
%
% NO OUTPUT


[d,n]=size(xTr);

% Feel free to change this code any way you want
f=@(w) ridge(w,xTr,yTr,0.1);
w0=zeros(size(xTr,1),1); % initialize with all-zeros vector
w=grdescent(f,w0,1e-06,5000);

save('w0','w');


