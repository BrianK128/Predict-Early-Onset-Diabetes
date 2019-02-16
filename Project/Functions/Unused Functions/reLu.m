function [xo] = reLu(x)
%reLu Leaky Rectified linear unit
%   Returns x if x >= 0, else 0.
    n = length(x);
    xo = zeros(n,1);
    for i = 1:n
        if(x(i) >= 0)
            xo(i) = x(i);
        else
            xo(i) = 0.01*x(i);
        end
    end
end

