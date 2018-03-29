%% Split data 

function [itr, ite] = valsplit(n,k,iter)

            splitsize = floor(n/k);

            ite = (1 + (iter-1) * splitsize):( iter * splitsize );
            itr = [1:( (iter-1) * splitsize ) ( iter * splitsize + 1 ):n];

end

% [d,n]=size(X);
% 
% split = ceil((1 - 1/k) * n);
% 
% itr=1:split;
% ite=split+1:n;
% 
% ii=randperm(length(Y));
% x=X(:,ii);
% y=Y(1,ii);
% 
% xTr=x(:,itr);
% yTr=y(itr);
% xTv=x(:,ite);
% yTv=y(ite);
% clear('itr', 'ite', 'n', 'd', 'x', 'y');