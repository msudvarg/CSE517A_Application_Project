%% Implements 10 fold cross validation and outputs mean -log predictive density 

%Inputs:  
%  x:   a NxD deisgn matrix
%  y:   Nx1  vector of response variables 
%  covfunc:  a cell or string listing the Covariance function to use 
%  hyp: a struct containing the hyperparamters 
%  seed:  seed for randomization to ensure CV cuts at same point each time

%Output: 
%  NLPD: average negative log predictive density of the ten fold cv 
%  stdNLPD: standard deviation 

function [meanNLPD,stdNLPD,PredMean,PredStd,hyp] = GP_crossval(x,y,covfunc,seed)

%Ensures using same random sequence
rng(seed);   
idx=randperm(length(y));

%Shuffling data
y=y(idx);
x(idx,:); 

%Splits 
SplitPoints=[1,ceil(length(y).*(.1:.1:1))];

%Pre-Allocate
NLPD=zeros(10,1); 
PredAccuracy=zeros(10,1); 


meanfunc=[]; 
likfunc=@likGauss; 

hyp=struct('mean',[], 'cov',[log(1/sqrt(2)),log(1)], 'lik', log(.5));
%hyp=struct('mean',[], 'cov',[log(1/sqrt(2)),log(1)]);




for i=1:10 
    xTr=x; 
    yTr=y;
    
    xTe=x(SplitPoints(i):SplitPoints(i+1),:);
    yTe=y(SplitPoints(i):SplitPoints(i+1));
    xTr(SplitPoints(i):SplitPoints(i+1),:)=[]; 
    yTr(SplitPoints(i):SplitPoints(i+1))=[];
        
    %Take random sample of training Set to optimze hyperparameters on     
    N=length(yTr); 
    uix=randsample(N,round(N/10,0),false); 
  
  %Sset a prior for numerical stability  
  mu=log(.5);
  s2=5;  
  prior.cov = {{@priorGauss,mu,s2}; { @priorGauss,mu,s2}};   
  prior.lik = {{@priorDelta}};
  inf = {@infPrior,@infGaussLik,prior};

    %Optimize Hyperparameters
     hyp=minimize(hyp, @gp, -100, inf, meanfunc, covfunc, likfunc, xTr(uix,:), yTr(uix));
    %hyp=minimize(hyp, @gp, -100, @infEP, meanfunc, covfunc, @likLogistic, xTr(uix,:), yTr(uix));
  
    
    %Train model on full Trainng Set 
    [y_pred_mu,y_pred_var] = gp(hyp, @infGaussLik, meanfunc,covfunc, likfunc, xTr, yTr,xTe);
    %[y_pred_mu,y_pred_var] = gp(hyp, @infEP, meanfunc,covfunc, @likLogistic, xTr, yTr,xTe);
    
    NLPD(i)= mean(.5.*log(2*pi.*y_pred_var) + (yTe-y_pred_mu).^2./(2.*y_pred_var)); 
    
    PredAccuracy(i)= sum(sign(y_pred_mu)==yTe)/length(yTe);


end

meanNLPD=mean(NLPD);
stdNLPD= std(NLPD); 
PredMean=mean(PredAccuracy);
PredStd=std(PredAccuracy); 


end

