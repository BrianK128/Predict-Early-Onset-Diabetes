function [proj] = pcProj(d,pc)
%PCPROJ Projects data using principal components. 
%   Returns data projected using the principal components. Dimensions equal
%   number of principal components given, with PC1 in the 1st dimension,
%   PC2 in 2nd dimension, etc. 
%   Data d should be the dataset normalized minus the sample mean. Each
%   column should represent an individual sample ie: [x1 x2 x3 ... xn]
%   (z columns for z samples)
%   for an n-dimensional sample. Each row should represent a feature. 
%   PC should be an n x m matrix where n is the number of initial classes
%   and m is the number of principal components chosen.
%   Returns a matrix with m x z rows where z is the number of samples in
%   the dataset. 

proj = zeros(size(pc,2),length(d));

for i = 1:length(d)
    for j = 1:size(pc,2)
        proj(j,i) = pc(:,j).'*d(:,i);
    end        
end
end

