%%CPE 646 
%Brian Kim
%This project uses PCA and K means clustering to determine if a patient
%is at risk of developing diabetes based on data from diagnostic tests.
%%
%
%%
%Load Data
dataSet = csvread('diabetes.csv',1,0);
%Since matlab csvread can only take in numeric values, the 1st row, the
%title row is cut off. Set string array to hold the values  of the 1st row.
title = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age", "Outcome"];
%Shortened Title String for scatter matrix.
shortTitle = ["P", "G", "BP", "ST", "I", "BMI", "DPF", "Age"];

%Variable meanings: 
%{ 
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration at 2 hours in an oral glucose tolerance test
Blood Pressure: Diastolic blood pressure (mm Hg)
Skin Thickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
%}
%%
%Split data set into training set and test set. Visualize and inspect data.
%Get number of rows in data set. Use 90% for training, 10% for testing.
nRows = size(dataSet,1);
trRow = ceil(.9*nRows);

%Training set. Training set is 90% of data set.
trSet = dataSet(1:trRow,:);
%test set is remaining 10%. Remove last row which we will use to classify.  
teSet = dataSet(trRow+1:end,1:8);

%Remove col 4 and 5. Change line for error column. (Line 49)
%trSet = [trSet(:,(1:3)) trSet(:,(6:9))];
%teSet = [teSet(:,(1:3)) teSet(:,(6:8))];
%% Initial visual analysis of the training set.

%histPlot(trSet,title); 
%scatterMatr(trSet,shortTitle);

%%
%1's denote that there are missing values in the column, 0 otherwise.
errCol = [0 1 1 1 1 1 0 0 0]; 
%errCol =[0 1 1 1 0 0 0] ;
%Substitute missing feature values with the mean of the "good" values. 
trSetEst = estMissFeature(trSet,errCol); 

%Separate class tag from dataset.
TrNorm = (trSetEst(:,(1:(size(trSetEst,2)-1))));
TrClass = trSetEst(:,(size(trSetEst,2)));

%Normalize and subtract mean from data set. 
TrNorm = norm_mean(TrNorm);

%% PCA
%Calculate Scatter matrix S.
S = 0;
for i = 1:length(TrNorm)
    temp = TrNorm(:,i) * TrNorm(:,i).';
    S = temp + S;
end

%Get eigen vectors and eigen values.
[vec,L] = eig(S);

%Get Eigen vectors that correspond with largest eigen values. Desired
%number of PC is 3 for visualization. 4 is used for testing performance.  
n = 3;
PC = princComp(vec,L,n);
%Reconstruct data using principal components. 
projData = pcProj(TrNorm,PC);

%get transpose of projected data.
projData = projData.';
projData = [projData trSetEst(:,(size(trSetEst,2)))];

%Estimate the missing feature values to be the mean of the good sample
%values.
teSetEst = estMissFeature(teSet,errCol);
%Normalize test set and subtract the sample mean. 
teNorm = norm_mean(teSetEst);

%Transform test data set into PC space.
projTest = pcProj(teNorm,PC);
projTest = projTest.'; 

%% SVM
%Train SVM Model
SVMModel = fitcsvm(TrNorm.',TrClass);

%Train SVM Model with PCA data.
PCA_SVM_Model = fitcsvm(projData(:,(1:size(PC,2))),TrClass);

%SVM gaussian function kernel
SVMKernMod = fitcsvm(trSetEst(:,1:(size(trSetEst,2)-1)),trSetEst(:,size(trSetEst,2)),...
'Standardize',true,'KernelFunction','gaussian','KernelScale','auto');

%% Train neural net
hLayer = 4;
NN1 = NN(size(TrNorm,1),hLayer);
r = .1;
T = TrNorm.';
%1 for rectified linear unit, other for logistic sigmoid
%Train neural net
epoch = 100;
for ep = 1:epoch
    NN1 = NN1.train(T,TrClass,r);
end
            

%% Classification
%Classify test set using k-nn 
knnClassifiedSet = twoC_knn(projData,projTest);

teAct = dataSet(trRow+1:end,9);

%Classify test set using SVM, SVM with PCA data
[label,~] = predict(SVMModel,teNorm.');
[label_PCA,~] = predict(PCA_SVM_Model,projTest);

[label_rbf,~] = predict(SVMKernMod,teSetEst);

%Classify using neural net.
LabNN_Norm = NN1.classify(teNorm.');
[labFit] = NN1.classify(T);

%% Determine Accuracy
%Determine accuracy of classifier using PCA and knn 
Acc_KNN_PCA = 1 - sum(abs(teAct - knnClassifiedSet(:,(n+1))))/length(teAct);

%Determine accuracy of SVM with normalized data set
Acc_SVM = 1 - sum(abs(teAct - label))/length(teAct);

%SVM accuracy with kernel function.
Acc_SVM_Norm_Kern = 1 - sum(abs(teAct - label_rbf))/length(teAct);

%Determine accuracy of SVM with dataset transformed using PCA.
Acc_SVM_PCA = 1 - sum(abs(teAct - label_PCA))/length(teAct);

%Determine accuracy of NN using normalized data set.
Acc_NN_Norm = 1 - sum(abs(teAct - LabNN_Norm))/length(teAct);
Acc_NN_Fit = 1 - sum(abs(TrClass - labFit))/length(TrClass);

%% Plot
%PCA plot 3D
%{
pcDataLab1 = 'Positive Diagnosis';
pcDataLab2 = 'Negative Diagnosis';
pcTitle = 'Training Set Principal Component Plot';
pcPlot3D(projData,n,pcTitle,pcDataLab1,pcDataLab2);
%}