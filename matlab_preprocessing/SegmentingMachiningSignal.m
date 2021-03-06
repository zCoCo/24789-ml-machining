function [Brass,Pmma,index]=SegmentingMachiningSignal(ChanNum,SpindleRPM, DAQ_IR ,NumberOfCycle,Feed,DAQ_Feed)

Force_axis=["AEraw","forcedatafx","forcedatafy","forcedatafz","IR","Microphone"];
ChannelNo=["First","Second","Third","Fourth"];
Numberr=ChanNum;
cur_dir=pwd;


Offset_index = (0.5/Feed)*DAQ_Feed; %% 0.5 mm of machining is removed from both ends

%NumberOfCycle=101;

NumOfDataForCycle=(DAQ_IR/(SpindleRPM/60))+10; %%Number of data points for a single rotation + 10 data points in case of rotation error!!..!!


%% LOAD IR DATA AE DATA and MIC

Force_axis=["AEraw","forcedatafx","forcedatafy","forcedatafz","IR","Microphone"];
cur_dir=pwd;


i=5;
dir = strcat(pwd ,'\Outputs\',num2str(Numberr),'\', Force_axis(i) , num2str(Numberr), '.txt')
res_dir=strcat(dir);
IR=textread(res_dir); 

i=1;
dir = strcat(pwd ,'\Outputs\',num2str(Numberr),'\', Force_axis(i) , num2str(Numberr), '.txt')
res_dir=strcat(dir);
AE=textread(res_dir); 

i=6;
dir = strcat(pwd ,'\Outputs\',num2str(Numberr),'\', Force_axis(i) , num2str(Numberr), '.txt')
res_dir=strcat(dir);
Mic=textread(res_dir); 

%% LOAD FORCE DATA
for i=2:4
    dir = strcat(pwd ,'\Outputs\',num2str(Numberr),'\', Force_axis(i) , num2str(Numberr), '.txt')
    res_dir=strcat(dir);
    data(:,i-1)=textread(res_dir); 
end

% %% PRINT IR SENSOR
% figure(99);
% plot(IR)

%% DETREND FORCE DATA BY BUILT_IN FUNCTION

Detrend_Force(:,1)=detrend(data(:,1));
Detrend_Force(:,2)=detrend(data(:,2));
Detrend_Force(:,3)=detrend(data(:,3));

%% PRINT THE DETRENDED FORCE DATA
for i=1:3
    
    figure()
    plot(Detrend_Force(:,i))
    hline = refline(0, 0);
    hline.Color = 'k';

end

%% USE NOISY DATA IN THE BEGINNING and LOOK AT SIGNAL TO NOIS RATIO FOR MOVING WINDOW AND THE NOISY DATA IN THE VERY BEGINNING

startsearch=1e4; %10 000 is the moving window size
stepsInIndex=startsearch;



j=1;
for i=startsearch:stepsInIndex:length(Detrend_Force) %divide the region 50 equal points
    
    
    if(length(Detrend_Force) - i < stepsInIndex)  %%% TO MAKE THE CODE MORE ROBOUST 
        break
    
    else
    r=snr(Detrend_Force(i:i+stepsInIndex-1,2),Detrend_Force(1:startsearch,2)); %SIGNAL TO NOISE RATIO
    end
    
    if( r > 5.00) %% IT SEEMS TO BE WORKING !!!! NEEDS TO NE CHANGED 
        
        %if(index(j,2)+1 ~= i)
            
            index(j,1)=i;
            index(j,2)=i+stepsInIndex-1;
            index(j,3)=r;
            %figure()
            %plot(Detrend_Force(i:i+stepsInIndex,2))
            
            j=j+1; %Record number of windows having data.
            
        %end
    end 
    
end

ChannelNum=1; %% RECORD START AND END INDICES OF 8 CHANNELS = 4 BRASS 4 PMMA!!!
i=2;
channelsRoughIndex=zeros(4,2);
while(i<length(index))
    channelsRoughIndex(ChannelNum,1)=index(i-1,1);
    while(index(i,1)==index(i-1,2)+1)
        i=i+1;
        if(i>length(index))
            break
        end
    end
    channelsRoughIndex(ChannelNum,2)=index(i-1,2);
    ChannelNum=ChannelNum+1;
    i=i+1;
 
end
[numRows,numCols] = size(channelsRoughIndex);
Offsetarray=ones(numRows,1)*Offset_index;
channelsRoughIndex(:,1)=channelsRoughIndex(:,1)+Offsetarray;
channelsRoughIndex(:,2)=channelsRoughIndex(:,2)-Offsetarray;


%% SAVE and PLOT SEGMENTED PORTIONS

for i=1:1:length(channelsRoughIndex)    
%    figure()
%    plot(Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),2))
%    
if( length(IR)<channelsRoughIndex(i,2)*4) %sometimes IR AE sensor card is interepted before 4th channel.
    channelsRoughIndex(i,2)=length(IR)/4;
    if(length(IR)<channelsRoughIndex(i,1)*4 )
        break
    end
end
   if (mod(i,2)==1)
           
       
      Brass{ceil(i/2),1}.Fx=Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),1);
      Brass{ceil(i/2),1}.Fy=Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),2);
      Brass{ceil(i/2),1}.Fz=Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),3);
      Brass{ceil(i/2),1}.Mic= Mic(channelsRoughIndex(i,1):channelsRoughIndex(i,2));
      Brass{ceil(i/2),1}.IR=IR(channelsRoughIndex(i,1)*4:channelsRoughIndex(i,2)*4); %They have 4 times higher acqusition rates...
      Brass{ceil(i/2),1}.AE=AE(channelsRoughIndex(i,1)*4:channelsRoughIndex(i,2)*4) ;%
      Brass{ceil(i/2),1}.index=channelsRoughIndex(i,1):channelsRoughIndex(i,2);
      [ForceinitialIndex,IRinitialIndex] = FindIRstart(IR(channelsRoughIndex(i,1)*4:channelsRoughIndex(i,2)*4),NumOfDataForCycle);
      Brass{ceil(i/2),1}.rotationstart(1)=ForceinitialIndex;  %% for 125khz   force,mic,..
      Brass{ceil(i/2),1}.rotationstart(2)=IRinitialIndex;  %% for 500khz  AErmse

      
   else
     
      Pmma{i/2,1}.Fx=Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),1);
      Pmma{i/2,1}.Fy=Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),2);
      Pmma{i/2,1}.Fz=Detrend_Force(channelsRoughIndex(i,1):channelsRoughIndex(i,2),3);
      Pmma{i/2,1}.Mic= Mic(channelsRoughIndex(i,1):channelsRoughIndex(i,2));
      Pmma{i/2,1}.IR=IR(channelsRoughIndex(i,1)*4:channelsRoughIndex(i,2)*4); %They have 4 times higher acqusition rates...
      Pmma{i/2,1}.AE=AE(channelsRoughIndex(i,1)*4:channelsRoughIndex(i,2)*4) ;%
      Pmma{i/2,1}.index=channelsRoughIndex(i,1):channelsRoughIndex(i,2);
      [ForceinitialIndex,IRinitialIndex] = FindIRstart(IR(channelsRoughIndex(i,1)*4:channelsRoughIndex(i,2)*4),NumOfDataForCycle);
      Pmma{ceil(i/2),1}.rotationstart(1)=ForceinitialIndex;  %% for 125khz   force,mic,..
      Pmma{ceil(i/2),1}.rotationstart(2)=IRinitialIndex;  %% for 500khz  AErmse
      
   end
end

%%
%%% Now start with IR analysis!!

% ForceinitialIndex,IRinitialIndex = FindIRstart(IR,NumOfDataForCycle);



end


