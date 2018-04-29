results = zeros(10,5);

%% Pre-Process data 

%Load data
BarackObama = readtable('twitter-ratio\BarackObama.csv');
realDonaldTrump = readtable('twitter-ratio\realDonaldTrump.csv');

Data=vertcat(BarackObama, realDonaldTrump); 

textData = Data.text;
labels = Data.user; 

%Removes punctuation and make lowercase
cleanTextData = lower(textData);
%cleanTextData = erasePunctuation(cleanTextData);
cleanTextData = regexprep(cleanTextData, 'http.+', ' ');

%Tokenize documents
cleanDocuments = tokenizedDocument(cleanTextData);

%Get rid of stopwords 
%cleanDocuments = removeWords(cleanDocuments,stopWords);

%
%cleanDocuments = removeShortWords(cleanDocuments,2);
%cleanDocuments = removeLongWords(cleanDocuments,15);

%Uses the Porter Stemmer algorithm 
cleanDocuments = normalizeWords(cleanDocuments);

%% Preprocessing for removal of words appearing 2 or fewer times

%Create Bag-of-Words
cleanBag = bagOfWords(cleanDocuments);
cleanBag = removeInfrequentWords(cleanBag,2);

%Getrid of empty docs 
[cleanBag,idx] = removeEmptyDocuments(cleanBag);
labels(idx) = [];

tfidfBag=tfidf(cleanBag);

%Convert labels to -1/+1, where +1 is Trump, -1 is Obama
labels = categorical(labels);
Y = double(labels);
Y(Y==1) = -1;
Y(Y==2) = 1;
Y = Y';
X = tfidfBag';

%% Gaussian Process Model
GP_Control

%% Preprocessing experiment for removal of words appearing 2 to 20 or fewer times

for freq_threshold = 2:2:20

    %Create Bag-of-Words
    cleanBag = bagOfWords(cleanDocuments);
    cleanBag = removeInfrequentWords(cleanBag,freq_threshold);

    %Getrid of empty docs 
    [cleanBag,idx] = removeEmptyDocuments(cleanBag);
    labels(idx) = [];

    tfidfBag=tfidf(cleanBag);

    %Convert labels to -1/+1, where +1 is Trump, -1 is Obama
    labels = categorical(labels);
    Y = double(labels);
    Y(Y==1) = -1;
    Y(Y==2) = 1;
    Y = Y';
    X = tfidfBag';

    %Perform k-fold cross validation
    k = 10;
    aucTotal = 0;
    errTreeTotal = 0;

    s = RandStream('mt19937ar','Seed',2018);
    ii=randperm(s,length(Y));
    X=X(:,ii);
    Y=Y(1,ii);
    [d,n]=size(X);

    %% Train linear model using ridge regression
    tic
    for i = 1:k
        [itr,ite] = valsplit(n,k,i);
        xTr=X(:,itr);
        yTr=Y(itr);
        xTv=X(:,ite);
        yTv=Y(ite);

        traintwitter(xTr,yTr);
        [fpr,tpr,auc] = isTrump(xTv,yTv);
        aucTotal = aucTotal + auc;

    end
    Linear_AUC = aucTotal/k
    toc

    %% Decision tree
    tic
    for i = 1:k
        [itr,ite] = valsplit(n,k,i);
        xTr=X(:,itr);
        yTr=Y(itr);
        xTv=X(:,ite);
        yTv=Y(ite);

        xTrTree = full(xTr');
        xTvTree = full(xTv');
        yTrTree = yTr';
        yTvTree = yTv';

        tree = fitctree(xTrTree, yTrTree);

        yTree = predict(tree,xTvTree);
        errTree = sum(yTree ~= yTvTree)/size(yTree,1);
        errTreeTotal = errTreeTotal + errTree;

    end
    Decision_Tree_Error = errTreeTotal/k
    toc


    %% PCA
    tic
        for i=1:10
        warning('off','all');
        [coeff, score, latent, tsquared, explained] = pca(full(X));
        warning('on','all');
        end
    pcatime = toc/10;
    
    %%
    x = 1:length(latent);

    figure();
    plot(x,latent);
    title('PCA Variances');
    xlabel('Principal Component');
    ylabel('Eigenvalue');

    figure();
    plot(x,cumsum(explained));
    title('Cumulative Variance by Principal Component');
    xlabel('Principal Component');
    ylabel('Cumulative Variance (%)');

    pcacount = [0 0];
    per10 = sum(latent > (latent(1)/10))
    pcacount(1,1) = per10;
    e = cumsum(explained);
    per90 = sum(e<90)
    pcacount(1,2) = per90;
    
    results(freq_threshold/2, :) = [ size(X,1) (sum(latent > (latent(1)/10))) (sum(e<90)) Linear_AUC pcatime];
    
    %%
    for c = pcacount
        X_PCA = (coeff(:,1:c))'; 
        
        aucTotal = 0;    
        tic
        for i = 1:k
            [itr,ite] = valsplit(n,k,i);
            xTr=X_PCA(:,itr);
            yTr=Y(itr);
            xTv=X_PCA(:,ite);
            yTv=Y(ite);

            traintwitter(xTr,yTr);
            [fpr,tpr,auc] = isTrump(xTv,yTv);
            aucTotal = aucTotal + auc;

        end
        Linear_AUC = aucTotal/k
        toc
          
        errTreeTotal = 0;
        tic
        for i = 1:k
            [itr,ite] = valsplit(n,k,i);
            xTr=X_PCA(:,itr);
            yTr=Y(itr);
            xTv=X_PCA(:,ite);
            yTv=Y(ite);

            xTrTree = full(xTr');
            xTvTree = full(xTv');
            yTrTree = yTr';
            yTvTree = yTv';

            tree = fitctree(xTrTree, yTrTree);

            yTree = predict(tree,xTvTree);
            errTree = sum(yTree ~= yTvTree)/size(yTree,1);
            errTreeTotal = errTreeTotal + errTree;

        end
        Decision_Tree_Error = errTreeTotal/k
        toc
    end
end

%% PCA Runtime Analysis

    figure();
    plot(results(:,1),results(:,5));
    t = (results(:,1).^2*6439 + results(:,1).^3)*3*10^-10;
    hold on;
    plot(results(:,1),t);
    title('PCA Runtime');
    xlabel('Dataset Dimensionality');
    ylabel('PCA Runtime (s)');
    legend('Mean Runtime','Theoretical Runtime');
