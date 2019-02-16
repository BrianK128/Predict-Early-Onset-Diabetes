function [cD] = twoC_knn(trD,teD)
%twoC_knn classifies a test dataset using k-nearest neighbors. This
%function is for a two class binary classifier. This function uses euclidean
%distance to calculate the nearest neighbors. The data set should be
%normalized to ensure proper performance. k is equal to the square root of
%the number of samples in the training data set. 
%   Return values are:
%   cD: classified test data set. 
%   The return format will be: Columns represent a feature. The last column
%   is the class label. Each Row is a sample.
%   Inputs to the function are:
%   trD: training dataset 
%   teD: test data set or unknown samples to be classified. 
%   **Input dataset should be sorted in the following manner:
%   Columns should represent a feature value. 
%   The last column of the training data set should be the class label for 
%   the test sample. 
%   Each row should be a sample. 
    
    cD = teD;
    n = length(trD);
    m = length(teD);
    %Number of features.
    f = size(trD,2);
    %Class labels
    C1 = min(trD(:,f));
    C2 = max(trD(:,f));
    
    %k is equal to the square root of n, rounded up to the nearest odd
    %integer.
    k = 2*floor(sqrt(n)/2)+1; 

    %Training data with class column removed.
    tDnC = trD(:,(1:(f-1)));
    
    %For every sample in the test dataset..
    for i = 1:m
        %Get the Euclidean distance between test sample m and training
        %set samples.
        E = sqrt(sum(((tDnC - teD(i,:)).^2).').');
        %initialize count for neighbors of each class for each test 
        %sample to 0.
        c1k = 0;
        c2k = 0;
        
        %Add class label to each euclidean distance between the mth test
        %sample and the training dataset.
        EC = [E trD(:,f)];
        %For k nearest neighbors, count the number of neighbors belonging
        %to each class.
        for j = 1:k
           [~,r] = min(EC(:,1));
           %increment count for each class depending on which class the
           %next neighbor is in.
           if(EC(r,2) == C1)
                c1k = c1k + 1;
           elseif(EC(r,2) == C2)
                c2k = c2k + 1;
           else
                disp('error. There are more than 2 classes in the training dataset for knn');
               
           end
           %Remove the nearest neighbor and find the next closest neighbor.
           EC(r,:) = [];    
        end
        %Set the class of the test sample to whichever has more neighbors.
        if (c1k > c2k)
            cD(i,f) = C1;
        else
            cD(i,f) = C2;
        end
    end
end

