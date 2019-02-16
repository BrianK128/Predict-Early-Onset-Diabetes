function [PC] = princComp(vec,L,n)
%PRINCCOMP Returns the requested Principal Components.
%   This function returns a column-wise matrix with the requested principal 
%   components given d x d matrices for eigen vectors and eigen values and 
%    the number of desired Principal Components. Each column is a principal
%    component where the 1st column is the 1st principal component, 2nd is
%    the 2nd principal component, etc. 


%Turn eigen value matrix into a single row, then select the eigen values
%that are the largest. The corresponding eigen vectors will be the 
%principal components.
Lrow = max(L);
Lmax = 0;
PC = zeros(length(L),n);
for i = 1:n
    if i == 1
        [Lmax,c] = max(Lrow);
        PC(:,1) = vec(:,c);
    else
   [Lmax,c] = max(Lrow(Lrow<Lmax));
   PC(:,i) = vec(:,c);
    end
end

end

