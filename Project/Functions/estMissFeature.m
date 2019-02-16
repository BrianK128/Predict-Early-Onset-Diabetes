function [dEst] = estMissFeature(d,e)
%ESTMISSFEATURE Estimates missing feature values using existing sample
%mean.
%Finds the mean for each feature (not including 0 values and sets
%the missing value that mean. 
%d: the training data set with missing features.
%m: a row vector indicating whether or not the 0's in a given feature is
%a missing value and not a valid value. (ie: it is possible for the number
%of pregnancies to be 0, but it is not possible for BMI to be 0.)
%dEst: the training dataset with the invalid 0s replaced with the mean of
%the good feature values.
%The Dataset input should be in the form of: 
%Rows: Each row represents a sample value
%Columns: Each column represents a feature in the input.
%The resulting output is in the same form.
%%
%Get number of non-zero values in every column.
idx = d~=0;
c = sum(idx,1);
m = sum(d);
dEst = d;
%If it is an error column, replace zero values with the mean. 
for i = 1:length(e)
    if(e(i) == 1)
        t = m(i)/c(i);
        for j = 1:length(d)
            if(dEst(j,i) == 0)
                dEst(j,i) = t; 
                else%Do Nothing
            end
        end
    else%Do Nothing.
    end        
end
end

