function [xo] = logSig(x)
%LOGSIG Logistic Sigmoid function.
%   Implements the function y = exp(x)/( 1 + exp(x) )
    n = length(x);
    xo = zeros(n,1);
    for i = 1:n
        xo(i) = exp(x(i)) / (1 + exp(x(i)));
    end
end

