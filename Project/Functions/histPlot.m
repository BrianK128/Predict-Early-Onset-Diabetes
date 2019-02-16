function [] = histPlot(d,t)
%HISTPLOT Plots a histogram for a data set given the data set and a row
%vector with the title of its corresponding column.
%Each row should represent an individual training sample. 
L = size(d,2);

neg = d(d(:,L) == 0,:);
pos = d(d(:,L) == 1,:);

for i = 1:(L-1)
   figure(i),histogram(neg(:,i),20);
   hold on
   figure(i),histogram(pos(:,i),20);
   title(t(i));
   hold off
   legend('Negative Diagnosis','Positive Diagnosis');
end
figure(L),histogram(d(:,L));
title(t(L));
legend('Diagnosis: 0 = Negative, 1 = Positive');

end

