clc
clear all
close all

%% TO DO LIST %%
% t-test to differentiate small postion of enterance and no machining
% region
% Match IR signal with Force and generated the desired time-length as a (2.5mm - 5mm )
% discreet time step
%
%% 2 - 3- 4 , 4 fails because of missing IR/AE data , there is multiple of those failures so need a check.
Feed=10;
DAQ_Feed=125000;
%% it is giving an error at 4th run!!
RunNumber=2
for i=4:4
[BRS,PMMA]=SegmentingMachiningSignal(i,6e4, 5e5,51,Feed,DAQ_Feed);


save_filename = strcat('Run',num2str(num2str(i)),'Brass'); %%% RENAME IT
save(save_filename,'BRS');
% See .mat file
save_filename = strcat('Run',num2str(num2str(i)),'Pmma'); %%% RENAME IT
save(save_filename,'PMMA');

end


%% Roughly 1 sec of Brass machining and 0.4 Sec PMMA


% figure()
%     plot(Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),2))
%     plot()
    MachLength = length(BRS{1,1}.Fx)/10; %% 0.1 sec
    
  %%% 0.3 sec -- 0.4 sec %%%
  
%for k=1:1:1
    
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

%%

Feed=10;
DAQ_Feed=125000;
[BRS_offset,PMMA]=SegmentingMachiningSignal(4,6e4, 5e5,51,Feed,DAQ_Feed);
%%

figure()
plot(BRS{4,1}.Fy)
figure()
plot(BRS{4,1}.IR)
%%

% numOfrotation=125
% [ForceinitialIndex,IRinitialIndex] = FindIRstart(BRS_offset{1,1}.IR,numOfrotation)


%%

    
    
