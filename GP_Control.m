%% Apply PCA to Data


A=flipud(eig(cov(X')));
plot(A/sum(A))

%X=full(X');
X_GP=full(X');

[coeffm score,latent]=pca(X_GP);

X_transformed=score(:,1:5); 


%% Run a stanrd SEL Kernel 
covfunc= @covSEiso; 

seed=2018; 
tic
[meanNLPD_SEL,stdNLPLD_SEL,AccMean_SEL,AccStd_SEL,hyp_SEL]=  GP_crossval(X_transformed,Y',covfunc,seed); 
toc


%% Matern Kernel with 3/2 degrees of freedom 

covfunc= {@covMaterniso,3}; 
seed=2018;

[meanNLPD_Matern3,stdNLPLD_Matern3,AccMean_Mater3,AccStd_Matern3,hyp_Matern3]=  GP_crossval(X_transformed,Y',covfunc,seed); 



%% %% Matern Kernel with 1/2 degrees of freedom 

covfunc= {@covMaterniso,1}; 
seed=2018;

[meanNLPD_Matern1,stdNLPLD_Matern1,AccMean_Matern1,AccStd_Matern1,hyp_Matern1]=  GP_crossval(X_GP,Y',covfunc,seed); 






