function [N] = norm_mean(d)
%NORM_MEAN Returns the mean-normalized data set. 
%   This function first normalizes the data set "d", then subtracts from 
%   it its mean. 
%   The format of the input data set should be:
%   Row: Each row represents an individual sample.
%   Column: Each column represents an individual feature value.
%   Each subsequent column is another dimension of the feature value.
%   Input format is in the form: 
%   Rows: Each row is a unique sample.
%   Columns: Each column represents a unique feature.
%   Output format:
%   Rows: Each row represents a unique feature.
%   Column: Each column represents a unique output.
%%    
    N = (d - min(d)) ./(max(d) - min(d));
    m1 = mean(N);
    N = N - m1;
    N = N.';


end

