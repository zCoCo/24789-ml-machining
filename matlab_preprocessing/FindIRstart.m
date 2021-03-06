function [ForceinitialIndex,IRinitialIndex] = FindIRstart(IR,numOfrotation)
%%
%plot for debugging
figure()
plot(IR)
%%

%numOfrotation=numOfrotation*4; % there is more data points at 500khz acqusition rate.
%We need to check out IR sensor to make sure that we can find cycle start location
% 1000rps at 500,000 =500 IR data
[peak_disp, peak_index] = max(IR(numOfrotation:2*numOfrotation)); 
%Find the first peak after roughsearch initial point

dummyCycleStartInd =  numOfrotation + peak_index - 20; 
%%Move 10 index back to the first peak in IR data cycle.

FindIRstartIndex=diff(IR(dummyCycleStartInd:dummyCycleStartInd+20));
%%Look at the differences in those 10 points
move=0;
for i=1:length(FindIRstartIndex)
    if (FindIRstartIndex(i)>0.1)
        move=i;
        break
    end
end

IRinitialIndex = (dummyCycleStartInd-1)+move;


%[peak_disp, peak_index] = max(FindIRstartIndex);
%%Find the maximum difference and its index among 10 points 
% 
% if(peak_index<1.0)  %% I do not remember why it is implemented!!
%    peak_index=peak_index+1; 
% end

%IRinitialIndex = (dummyCycleStartInd-1)+peak_index;
ForceinitialIndex = floor(IRinitialIndex/4);
end




