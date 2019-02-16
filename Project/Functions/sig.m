function [y] = sig(x)
%SIG Sigmoid activation function.
    
    n = length(x);
    y = zeros(n,1);
    for i = 1:n
        y(i) = 1 / (1 + exp(-x(i)));
    
    end
end

