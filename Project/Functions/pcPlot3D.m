function [] = pcPlot3D(projData,n,t,dLab1,dLab2)
%PCPLOT3D 3D plot using first three principal components (Or any three dim)
%   Implemented as separate function to reduce clutter in main script.

hD = projData(projData(:,n+1) == 1,:);
nD = projData(projData(:,n+1) == 0,:);

hold on

scatter3(hD(:,1),hD(:,2),hD(:,3),'MarkerEdgeColor','k',...
                                       'MarkerFaceColor',[1 0 0]),view(-60,60);

scatter3(nD(:,1),nD(:,2),nD(:,3),'MarkerEdgeColor','k',...
                                'MarkerFaceColor',[0 0 1]);

xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend(dLab1,dLab2);
title(t);
hold off           
end

