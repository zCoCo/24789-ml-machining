 # -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:38:46 2021

@author: alialp
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, C_AE_Feature, C_MIC_Feature,  C_FORCE_Feature, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
#%% Convolution Layer for the ACOUSTIC EMISSION

        #start with 37501 points
        self.CNN_AE_features = nn.Sequential(
                   
            nn.Conv1d(1, 3, 61, stride=20),   #1873 points in 3 channels
            nn.BatchNorm1d(3), #?? What this value should be
            nn.ReLU(),
            nn.Conv1d(3, 5, 23 , stride=10), # 186 points in 5 channels
            nn.ReLU(),
            nn.BatchNorm1d(5),
       
               
            nn.Conv1d(5, 8, 4 , stride=2), #92 points in 8 channels 
            nn.ReLU(), 
       
            nn.AvgPool1d(kernel_size=4,stride=2), # 45 points averaged!!

        # self.Dense1=nn.Linear(12*18,120)  # Number of channels x Points in the channels if no Avgpool is applied. 
        # self.Dense1Act=nn.ReLU()

            nn.Linear(45,C_AE_Feature),  # Number of channels x Points in the channels if no Avgpool is applied. 
            nn.ReLU()
        )
#%% Convolutional Layer For Microphone

        self.CNN_Mic_features = nn.Sequential(
            #start with 9376 points
            nn.Conv1d(1, 3, 46, stride=10),   #934 points in 3 channels
            nn.BatchNorm1d(3), #?? What this value should be
            nn.ReLU(),
            nn.Conv1d(3, 5, 14 , stride=5),# 186 points in 5 channels
            nn.ReLU(),
            nn.BatchNorm1d(5),
       
               
            nn.Conv1d(5, 8, 3, stride=2), #92 points in 8 channels
            nn.ReLU(), 
       
            nn.AvgPool1d(kernel_size=4,stride=2), # 45 points averaged!!

        # self.Dense1=nn.Linear(12*18,120)  # Number of channels x Points in the channels if no Avgpool is applied. 
        # self.Dense1Act=nn.ReLU()

            nn.Linear(45,C_MIC_Feature),  # Number of channels x Points in the channels if no Avgpool is applied. 
            nn.ReLU()
        )
        
#%% CONVOLUTIONAL LAYER FOR FORCES
        self.CNN_Forces_features = nn.Sequential(
            #start with 9376 points
            nn.Conv1d(3, 6, 6, stride=2),   #92 points in 3 channels
            nn.BatchNorm1d(3), #?? What this value should be
            nn.ReLU(),
            nn.Conv1dnn.Conv1d(6, 15, 3 , stride=1), # 90 points in 5 channels
            nn.ReLU(),
            nn.BatchNorm1d(5),
       
               
            nn.Conv1d(15, 24, 4 , stride=2), #44 points in 8 channels 
            nn.ReLU(), 
       
            nn.AvgPool1d(kernel_size=4,stride=2), # 21 points averaged!!

            # THIS MIGHT BE MORE APPROPRIATE compared to Average pooling here but how to implement it ???
            # self.Dense1=nn.Linear(12*18,120)  # Number of channels x Points in the channels if no Avgpool is applied. 
            # self.Dense1Act=nn.ReLU()
            #output=output.view(-1,12*18) #out = out.view(out.size(0), -1) CHANNEL X NUMBER OF POINTS IN EACH CHANNEL IS  FED TO DENSE LAYER
            
            
            nn.Linear(21,C_FORCE_Feature),  # Number of channels x Points in the channels if no Avgpool is applied. 
            nn.ReLU()
        )
        
        self.input_size=C_FORCE_Feature+C_MIC_Feature+C_AE_Feature
        self.hidden_size=hidden_size=300
        self.num_layers=num_layers
        self.dropout=dropout
        
        self.lstmcell1=nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False, dropout= self.dropout)
        
        self.Dense1=nn.Linear(300,188)  #the final output should be 17 data points.
        self.Dense1Act=nn.Sigmoid()
        # self.Dense2=nn.Linear(34,17) #It generates 17 data points
#%%
        # lstm_input_size is determined by the number of features we get from self.features()
        # self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,num_layers=1, batch_first=True)
        # self.classifier = nn.Linear(32, 2)

        # self.optimizer = optim.Adam(self.parameters())
        # self.criterion = nn.NLLLoss()

    def forward(self, AE,Mic,Forces,tillTimeT):

        #  There will be sensor values from 1 to n and m number of features will be produced let`s say for LSTM, Therefore n x m 
        
        AE_2d_Features=torch.empty(tillTimeT,self.C_AE_Feature)
        Mic_2d_Features=torch.empty(tillTimeT,self.C_MIC_Feature)       
        Forces_2d_Features=torch.empty(tillTimeT,self.C_FORCE_Feature)  
        LSTMFeature=torch.empty(tillTimeT,) 
        
        for i in range(tillTimeT):
            AE_2d_Features[i,:]=self.CNN_AE_features(AE) # if T=20 , tensor dim = 20 X C_AE_Feature
            Mic_2d_Features[i,:]=self.CNN_Mic_features(Mic)  # if T=20 , tensor dim = 20 X C_MIC_Feature          
            Forces_2d_Features[i,:]=self.CNN_Forces_features(Forces) # if T=20 , tensor dim = 20 X C__Feature          
            LSTMFeature[i,:]=torch.cat((AE_2d_Features[i,:], Mic_2d_Features[i,:], Forces_2d_Features[i,:]), 1)

                
        # LSTMFeature=torch.cat((AE_2d_Features, Mic_2d_Features, Forces_2d_Features), 1)   That might work too.
        print(LSTMFeature.shape)
        LSTMFeature = LSTMFeature.unsqueeze(dim=0) # adds a 0-th dimension of size 1
        print(LSTMFeature.shape)
        
        output=self.lstmcell1(LSTMFeature)  # [batch_size, sequence_length, input_dim]
        OutputForceY=self.Dense1Act(self.Dense1(output))

        
        return OutputForceY