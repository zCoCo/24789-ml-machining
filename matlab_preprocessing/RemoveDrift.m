function [DataNew,startsearch]=RemoveDrift(data,startsearch)

figure(14)
plot(data);
title('Remove Drift from force data by hand-picked 12 points')
hold on
[x,y] = ginput(12);
x=round( x );
for i=1:1:12
offsetpointy(i)=data(x(i));
end

plotx=linspace(x(1),x(12));
[rn,mn,bn] = regression(x',offsetpointy);
Driftline =bn + mn*plotx;
hold on
plot(plotx,Driftline, 'LineWidth',3)
x_offset = [1:1:length(data)];
Substract=bn + mn*x_offset;
DataNew=data-Substract';
figure(15)
plot(DataNew)
%ylim([1.5 -2])

CuttingMaxForRoughSearch=max(DataNew(1:startsearch)); % when the mean of 1000 points is above that value for Fx stop index search
CuttingMinForRoughSearch=min(DataNew(1:startsearch)); % when the mean of 1000 points is above that value for Fx stop index search

while(1)
    MaxInSearch= max(DataNew(startsearch:startsearch+1000));
    MinInSearch= min(DataNew(startsearch:startsearch+1000));
    if(MaxInSearch> CuttingMaxForRoughSearch && MinInSearch<CuttingMinForRoughSearch)
       break 
    end
    startsearch=startsearch+1000;  
end


% while(FindChannel(DataNew,startsearch,CuttingMaxForRoughSearch,CuttingMinForRoughSearch)==0)
%    startsearch=startsearch+1000; 
% end
end