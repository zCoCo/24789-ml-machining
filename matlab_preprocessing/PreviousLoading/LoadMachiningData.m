function [ChannelCycle]=LoadMachiningData(ChanNum,SpindleRPM, DAQrate ,NumberOfCycle)

Force_axis=["AEraw","forcedatafx","forcedatafy","forcedatafz","IR","Microphone"];
ChannelNo=["First","Second","Third","Fourth"];
Numberr=0;
cur_dir=pwd;

%NumberOfCycle=101;

NumOfDataForCycle=(DAQrate/(SpindleRPM/60))+10; %%Number of data points for a single rotation + 10 data points in case of rotation error!!..!!

for i=2:4
 %Forcedatafx
dir = strcat(pwd ,'\Runs\', 'Run', num2str(ChanNum), Force_axis(i) , num2str(Numberr), '.txt')
res_dir=strcat(dir);
data1=textread(res_dir); 


% figure()
% plot(data1); %% You can commmend out this, it plots on REMOVEDRIFT funtion!!
RoughstartsearchOnForce= 2e5; %at 200k rate, it is the first second.
[dataCorr,InitialSearchLoc]=RemoveDrift(data1,RoughstartsearchOnForce);

%figure()
%plot(dataCorr(FirstLoc:FirstLoc+5e5)); it is to debug the starting point
% index for forces.

StartingIndexforForces(i-1)=InitialSearchLoc;
Forces(:,i-1)=dataCorr;

end


save_filename = strcat('NoDriftCh',num2str(ChanNum),'_',num2str(i-1)); %%% RENAME IT
save(save_filename,'Forces');

InitialSearchLoc=StartingIndexforForces(1)

%% IR sensor
% Find the value of IR just before a sharp increase, it will be the cycle start point
% 20cycle  =  60k = 1000RPS, 20mm/s , 0.02mm/R
%Numberr=5;

i=5;
dir = strcat(pwd ,'\Runs\', 'Run', num2str(ChanNum), Force_axis(i) , num2str(Numberr), '.txt')
res_dir=strcat(dir);
data2=textread(res_dir); 

SearchStartIndexOnIR=InitialSearchLoc+0.4e5; %0.2 sec move into forces to avoid entry region.
%0.4 was enough for run3/4 , InitialSearchLoc is a rough estimate on Fx forces.
%We need to check out IR sensor to make sure that we can find cycle start location

[peak_disp, peak_index] = max(data2(SearchStartIndexOnIR:SearchStartIndexOnIR+200)); 
%Find the first peak after roughsearch initial point
dummyCycleStartInd= (SearchStartIndexOnIR-1) + peak_index - 10; 
%%Move 10 index back to the first peak in IR data cycle.
FindIRstartIndex=diff(data2(dummyCycleStartInd:dummyCycleStartInd+10));
%%Look at the differences in those 10 points
[peak_disp, peak_index] = max(FindIRstartIndex);
%%Find the maximum difference and its index among 10 points 
if(peak_index<1.0)
   peak_index=peak_index+1; 
end

dummyCycleStartInd= (dummyCycleStartInd-1)+peak_index;
CycleStartInd(1)=dummyCycleStartInd;



for i=2:NumberOfCycle %% this portion take the difference values on IR sensor and finds the max location as a search starting of a slop increase.
    
FindIRstartIndex=diff(data2(CycleStartInd(i-1)+1:CycleStartInd(i-1)+NumOfDataForCycle));
[peak_disp, peak_index] = max(FindIRstartIndex);

if(peak_index<1.0) %IR values bigger than 4.00 when see black region on the tool
   peak_index=peak_index+1; 
end

CycleStartInd(i)= CycleStartInd(i-1) + peak_index;

CycleLength(i-1) = CycleStartInd(i)- CycleStartInd(i-1);

end


save_filename = strcat('CycleIndex',num2str(ChanNum),'_');
save(save_filename,'CycleStartInd');

%%
%
Aircut= 40 ;
AirFeed= 10 ;
MachningCut=30;
MachiningFeed=15;
IndexAdditionForNextChannel= ((Aircut/AirFeed)+MachningCut/MachiningFeed)*DAQrate;
%
ChannelCycle=cell(4,10);
for j=1:4
   % this does not make sure that we will find an exact location another search is needed. 
   NextChannelStartIndex = CycleStartInd(2)+ IndexAdditionForNextChannel;
   % when k is 11, then cycleStartInd is moved to the next channel
    
for k=1:11 % take 10 means of 100 machining flute cycle.
ChannelCycle{j,k}.Fx=zeros(min(CycleLength(2:end)),NumberOfCycle); %1 calculate as a average and 2 as STD
ChannelCycle{j,k}.Fy=zeros(min(CycleLength(2:end)),NumberOfCycle);
ChannelCycle{j,k}.Fz=zeros(min(CycleLength(2:end)),NumberOfCycle);
ChannelCycle{j,k}.CycleIndexes=CycleStartInd;
differenceInCycleTime=max(CycleLength)-min(CycleLength(2:end));


    
    
    
    for i=2:NumberOfCycle-1
    
        if (CycleLength(i)> min(CycleLength(2:end)) )
        DCT= CycleLength(i)-min(CycleLength(2:end)) ; %% Difference in Cycle Time
        ChannelCycle{j,k}.Fx(:,i)=Forces(CycleStartInd(i):CycleStartInd(i+1)-1-DCT,1);%change -1 with max min diff.
        ChannelCycle{j,k}.Fy(:,i)=Forces(CycleStartInd(i):CycleStartInd(i+1)-1-DCT,2);%change -1 with max min diff.
        ChannelCycle{j,k}.Fz(:,i)=Forces(CycleStartInd(i):CycleStartInd(i+1)-1-DCT,3);%change -1 with max min diff.

        else
       %%DCT= CycleLength(i)-min(CycleLength)-53 ; %% Difference in Cycle Time
        ChannelCycle{j,k}.Fx(:,i)=Forces(CycleStartInd(i):CycleStartInd(i+1)-1,1); %-DCT
        ChannelCycle{j,k}.Fy(:,i)=Forces(CycleStartInd(i):CycleStartInd(i+1)-1,2);
        ChannelCycle{j,k}.Fz(:,i)=Forces(CycleStartInd(i):CycleStartInd(i+1)-1,3);

        end
    
    end
    
    
    if(k==11) % when averaging of 10 - 100 cycles ends, we move to the next channel
        
    CycleStartInd(1)=NextChannelStartIndex;
   
    else
    CycleStartInd(1)=CycleStartInd(end); %% reset the cycle start by assining the final index to the first index
    end
    
    for i=2:NumberOfCycle %% this portion take the difference values on IR sensor and finds the max location as a search starting of a slop increase.

    FindIRstartIndex=diff(data2(CycleStartInd(i-1)+1:CycleStartInd(i-1)+NumOfDataForCycle));
    [peak_disp, peak_index] = max(FindIRstartIndex);

    if(peak_index<1.0) %IR values bigger than 4.00 when see black region on the tool
       peak_index=peak_index+1; 
    end

    CycleStartInd(i)= CycleStartInd(i-1) + peak_index;

    CycleLength(i-1) = CycleStartInd(i)- CycleStartInd(i-1);

    end   
    
end




end
%%














%%
% figure(199)
% [lineOut, fillOut] = stdshade(Channel1_20cycle1.Fx',0.2)
% hold on
% 
% figure(299)
% [lineOut, fillOut] = stdshade(Channel1_20cycle1.Fy',0.2)
% hold on
% 
% figure(399)
% [lineOut, fillOut] = stdshade(Channel1_20cycle1.Fz',0.2)
% hold on



end
