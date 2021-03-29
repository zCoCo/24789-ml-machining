clc
clear all
close all

Run5ChAll=LoadMachiningData(5,6e4, 2e5,101);


save_filename = strcat('Run5Channel1_10_100CyclesWmicAE'); %%% RENAME IT
save(save_filename,'Run5ChAll');
% See .mat file


%%

for k=1:10
figure(199)
[lineOut, fillOut] = stdshade(Run5ChAll{1,2}.Fx',0.2)
% hold on
xlim([0 205])
ylim([-1.2 1.0])

figure(299)
[lineOut, fillOut] = stdshade(Run5ChAll{1,2}.Fy',0.2)
% hold on
xlim([0 205])
ylim([-1.2 1.0])

figure(399)
[lineOut, fillOut] = stdshade(Run5ChAll{1,2}.Fz',0.2)
% hold on
xlim([0 205])
ylim([-1.2 1.0])

a=1
end

%%
clrForChannels=[1 0 0 0.2;0 1 0 0.2;0 0 1 0.2; 0 0 0 0.2];

for i=1:4
figure(199)
[lineOut, fillOut] = stdshade(Run5ChAll{i,5}.Fx',0.2)
lineOut.Color = clrForChannels(i,:);
% fillOut.Color=clrForChannels(i,:);
hold on
xlim([0 205])
ylim([-1.2 1.0])

figure(299)
[lineOut, fillOut] = stdshade(Run5ChAll{i,5}.Fy',0.2)
lineOut.Color = clrForChannels(i,:);

hold on
xlim([0 205])
ylim([-1.2 1.0])

figure(399)
[lineOut, fillOut] = stdshade(Run5ChAll{i,5}.Fz',0.2)
lineOut.Color = clrForChannels(i,:);

hold on
xlim([0 205])
ylim([-1.2 1.0])


end

%%
figure(111)
[lineOut, fillOut] = stdshade(Run5ChAll{1,2}.Min',0.2)
%%
for i=1:1:5
figure(112)
plot(Run5ChAll{1,2}.Mic(:,i)')

hold on
end


OneDArray = reshape(Run5ChAll{1,2}.Mic(:,2:end),1,[]);
%%
fftPlot(OneDArray,1,201,200000)

%%

for i=1:1:5
figure(112)
plot(Run5ChAll{1,2}.AE(:,i)')

hold on
end

OneDArrayAE = reshape(Run5ChAll{1,2}.AE(:,2:end),1,[]);
fftPlot(OneDArrayAE,1,20000,200000)
%%
fftPlot(OneDArrayAE,1,200,200000)



