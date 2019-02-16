function [] = scatterMatr(trSet,s)
%SCATTERMATR uses matlab function gplot to create a scatter matrix plot.
%   Created as a separate function to reduce clutter. 

figure

gplotmatrix(trSet(:,(1:8)),[],trSet(:,9),['b' 'r'],[],[],false);
text([.05 .18 .31 .42 .55 .68 .80 .92], repmat(-.1,1,8), s, 'FontSize',8);
text(repmat(-.12,1,8), [.93 .8 .68 .53 .42 .27 .15 .02], s, 'FontSize',8, 'Rotation',90);

end

