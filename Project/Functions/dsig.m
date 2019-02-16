function [y] = dsig(x)
%DSIG Derivative of sigmoid function
%Make sure that input is the resulting output of a sigmoid function.
    n = length(x);
    y = zeros(n,1);
    for i = 1:n
        y(i) = x(i) * (1 - x(i));
    end
end

