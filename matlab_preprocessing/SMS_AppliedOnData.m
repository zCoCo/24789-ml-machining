clc
clear all
close all
%% 2 - 3- 4 , 4 fails because of missing IR/AE data , there is multiple of those failures so need a check.
[BRS,PMMA]=SegmentingMachiningSignal(2,6e4, 5e5,51);
%% Roughly 1 sec of Brass machining and 0.4 Sec PMMA


% figure()
%     plot(Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),2))
%     plot()
    MachLength = length(BRS{1,1}.Fx)/10; %% 0.1 sec
    
  %%% 0.3 sec -- 0.4 sec %%%
  
for k=1:1:1
    
    figure(199)
    plot(BRS{1,1}.Fx(3*MachLength:4*MachLength))
    % hold on
    %xlim([0 125])
    ylim([-0.4 0.4])

    figure(299)
    plot(BRS{1,1}.Fy(3*MachLength:4*MachLength))
    % hold on
    %xlim([0 125])
    ylim([-0.4 0.4])

    figure(399)
    plot(BRS{1,1}.Fz(3*MachLength:4*MachLength))
    % hold on
    %xlim([0 125])
    ylim([-0.4 0.4])

    a=1
end
    
    
    
    
    
%     
% for k=1:1:1
%     figure(199)
%     [lineOut, fillOut] = stdshade(BRS{1,1}.Fx(3*MachLength:4*MachLength)',0.2)
%     % hold on
%     xlim([0 125])
%     ylim([-0.4 0.4])
% 
%     figure(299)
%     [lineOut, fillOut] = stdshade(BRS{1,1}.Fy(3*MachLength:4*MachLength)',0.2)
%     % hold on
%     xlim([0 125])
%     ylim([-0.4 0.4])
% 
%     figure(399)
%     [lineOut, fillOut] = stdshade(BRS{1,1}.Fz(3*MachLength:4*MachLength)',0.2)
%     % hold on
%     xlim([0 125])
%     ylim([-0.4 0.4])
% 
%     a=1
% end