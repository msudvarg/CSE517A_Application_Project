%% Pre-Process data 

%Load data
BarackObama = readtable('twitter-ratio\BarackObama.csv');
realDonaldTrump = readtable('twitter-ratio\realDonaldTrump.csv');

Data=vertcat(BarackObama, realDonaldTrump); 

textData = Data.text;
labels = Data.user; 

%Removes punctuation and make lowercase
cleanTextData = erasePunctuation(textData);
cleanTextData = lower(cleanTextData);
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

%Perform k-fold cross validation
k = 10;
aucTotal = 0;
errTreeTotal = 0;

for i = 1:k
    valsplit;
    
    %Train linear model using ridge regression
    traintwitter(xTr,yTr);
    [fpr,tpr,auc] = isTrump(xTv,yTv);
    aucTotal = aucTotal + auc;
    
    %Decision tree
    xTrTree = full(xTr');
    xTvTree = full(xTv');
    yTrTree = yTr';
    yTvTree = yTv';

    tree = fitctree(xTrTree, yTrTree);

    yTree = predict(tree,xTvTree);
    errTree = sum(yTree ~= yTvTree)/size(yTree,1);
    errTreeTotal = errTreeTotal + errTree;

end

aucTotal/k
errTreeTotal/k
