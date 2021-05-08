# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:38:46 2021

@author: alialp(crazy_boi_93@gmail.com)
edited(2021_05_08): yusuf mert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, CNN_AE_properties, CNN_MIC_properties, CNN_FORCE_properties, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        # %% Convolution Layer for the ACOUSTIC EMISSION
        self.CNN_AE_conv_layers = CNN_AE_properties[0]
        self.CNN_AE_conv_MaxPoolLayers = CNN_AE_properties[1]
        self.CNN_AE_conv_layer_activations = CNN_AE_properties[2]
        self.CNN_AE_conv_BN_layers = CNN_AE_properties[4]
        self.CNN_AE_conv_Dropout_layers = CNN_AE_properties[5]
        self.CNN_AE_linear_layers = CNN_AE_properties[6]
        self.CNN_AE_linear_activations = CNN_AE_properties[7]
        self.CNN_AE_linear_BN_layers = CNN_AE_properties[8]
        self.CNN_AE_linear_Dropout_layers = CNN_AE_properties[9]
        self.CNN_AE_modules = []

        self.CNN_MIC_conv_layers = CNN_MIC_properties[0]
        self.CNN_MIC_conv_MaxPoolLayers = CNN_MIC_properties[1]
        self.CNN_MIC_conv_layer_activations = CNN_MIC_properties[2]
        self.CNN_MIC_conv_BN_layers = CNN_MIC_properties[4]
        self.CNN_MIC_conv_Dropout_layers = CNN_MIC_properties[5]
        self.CNN_MIC_linear_layers = CNN_MIC_properties[6]
        self.CNN_MIC_linear_activations = CNN_MIC_properties[7]
        self.CNN_MIC_linear_BN_layers = CNN_MIC_properties[8]
        self.CNN_MIC_linear_Dropout_layers = CNN_MIC_properties[9]
        self.CNN_MIC_modules = []

        self.CNN_FORCE_conv_layers = CNN_FORCE_properties[0]
        self.CNN_FORCE_conv_MaxPoolLayers = CNN_FORCE_properties[1]
        self.CNN_FORCE_conv_layer_activations = CNN_FORCE_properties[2]
        self.CNN_FORCE_conv_BN_layers = CNN_FORCE_properties[4]
        self.CNN_FORCE_conv_Dropout_layers = CNN_FORCE_properties[5]
        self.CNN_FORCE_linear_layers = CNN_FORCE_properties[6]
        self.CNN_FORCE_linear_activations = CNN_FORCE_properties[7]
        self.CNN_FORCE_linear_BN_layers = CNN_FORCE_properties[8]
        self.CNN_FORCE_linear_Dropout_layers = CNN_FORCE_properties[9]
        self.CNN_FORCE_modules = []

        for index, i in enumerate(self.CNN_AE_conv_layers):
            self.CNN_AE_modules.append(nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_AE_conv_MaxPoolLayers[index] != False:
                self.CNN_AE_modules.append(nn.MaxPool1d(kernel_size=self.CNN_AE_conv_MaxPoolLayers[index][0],
                                                        stride=self.CNN_AE_conv_MaxPoolLayers[index][
                                                            0]))  # ?? What this value should be

            if self.CNN_AE_conv_layer_activations[index] == "sigmoid":
                self.CNN_AE_modules.append(nn.Sigmoid())
            elif self.CNN_AE_conv_layer_activations[index] == "relu":
                self.CNN_AE_modules.append(nn.ReLU())
            elif self.CNN_AE_conv_layer_activations[index] == "tanh":
                self.CNN_AE_modules.append(nn.Tanh())
            else:
                self.CNN_AE_modules.append(nn.Linear())

            if self.CNN_AE_conv_BN_layers[index] != False:
                self.CNN_AE_modules.append(nn.BatchNorm1d(CNN_AE_conv_BN_layers[index]))

            if self.CNN_AE_conv_Dropout_layers != False:
                self.CNN_AE_modules.append(nn.Dropout(CNN_AE_conv_Dropout_layers[index]))

        for index, i in enumerate(self.CNN_AE_linear_layers):
            self.CNN_AE_modules.append(nn.Linear(i[0], i[1]))

            if self.CNN_AE_linear_activations[index] == "sigmoid":
                self.CNN_AE_modules.append(nn.Sigmoid())
            elif self.CNN_AE_linear_activations[index] == "relu":
                self.CNN_AE_modules.append(nn.ReLU())
            elif self.CNN_AE_linear_activations[index] == "tanh":
                self.CNN_AE_modules.append(nn.Tanh())
            else:
                self.CNN_AE_modules.append(nn.Linear())

            if self.CNN_AE_linear_BN_layers[index] != False:
                self.CNN_AE_modules.append(nn.BatchNorm1d(CNN_AE_linear_BN_layers[index]))

            if self.CNN_AE_linear_Dropout_layers != False:
                self.CNN_AE_modules.append(nn.Dropout(CNN_AE_linear_Dropout_layers[index]))

        self.CNN_AE_features = nn.Sequential(self.CNN_AE_modules)

        for index, i in enumerate(self.CNN_MIC_conv_layers):
            self.CNN_MIC_modules.append(nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_MIC_conv_MaxPoolLayers[index] != False:
                self.CNN_MIC_modules.append(nn.MaxPool1d(kernel_size=self.CNN_MIC_conv_MaxPoolLayers[index][0],
                                                         stride=self.CNN_MIC_conv_MaxPoolLayers[index][
                                                             0]))  # ?? What this value should be

            if self.CNN_MIC_conv_layer_activations[index] == "sigmoid":
                self.CNN_MIC_modules.append(nn.Sigmoid())
            elif self.CNN_MIC_conv_layer_activations[index] == "relu":
                self.CNN_MIC_modules.append(nn.ReLU())
            elif self.CNN_MIC_conv_layer_activations[index] == "tanh":
                self.CNN_MIC_modules.append(nn.Tanh())
            else:
                self.CNN_MIC_modules.append(nn.Linear())

            if self.CNN_MIC_conv_BN_layers[index] != False:
                self.CNN_MIC_modules.append(nn.BatchNorm1d(CNN_MIC_conv_BN_layers[index]))

            if self.CNN_MIC_conv_Dropout_layers != False:
                self.CNN_MIC_modules.append(nn.Dropout(CNN_MIC_conv_Dropout_layers[index]))

        for index, i in enumerate(self.CNN_MIC_linear_layers):
            self.CNN_MIC_modules.append(nn.Linear(i[0], i[1]))

            if self.CNN_MIC_linear_activations[index] == "sigmoid":
                self.CNN_MIC_modules.append(nn.Sigmoid())
            elif self.CNN_MIC_linear_activations[index] == "relu":
                self.CNN_MIC_modules.append(nn.ReLU())
            elif self.CNN_MIC_linear_activations[index] == "tanh":
                self.CNN_MIC_modules.append(nn.Tanh())
            else:
                self.CNN_MIC_modules.append(nn.Linear())

            if self.CNN_MIC_linear_BN_layers[index] != False:
                self.CNN_MIC_modules.append(nn.BatchNorm1d(CNN_MIC_linear_BN_layers[index]))

            if self.CNN_MIC_linear_Dropout_layers != False:
                self.CNN_MIC_modules.append(nn.Dropout(CNN_MIC_linear_Dropout_layers[index]))

        self.CNN_Mic_features = nn.Sequential(self.CNN_MIC_modules)

        # %% CONVOLUTIONAL LAYER FOR FORCES
        for index, i in enumerate(self.CNN_FORCE_conv_layers):
            self.CNN_FORCE_modules.append(nn.Conv1d(i[0], i[1], i[2], stride=i[3]))
            if self.CNN_FORCE_conv_MaxPoolLayers[index] != False:
                self.CNN_FORCE_modules.append(nn.MaxPool1d(kernel_size=self.CNN_FORCE_conv_MaxPoolLayers[index][0],
                                                           stride=self.CNN_FORCE_conv_MaxPoolLayers[index][
                                                               0]))  # ?? What this value should be

            if self.CNN_FORCE_conv_layer_activations[index] == "sigmoid":
                self.CNN_FORCE_modules.append(nn.Sigmoid())
            elif self.CNN_FORCE_conv_layer_activations[index] == "relu":
                self.CNN_FORCE_modules.append(nn.ReLU())
            elif self.CNN_FORCE_conv_layer_activations[index] == "tanh":
                self.CNN_FORCE_modules.append(nn.Tanh())
            else:
                self.CNN_FORCE_modules.append(nn.Linear())

            if self.CNN_FORCE_conv_BN_layers[index] != False:
                self.CNN_FORCE_modules.append(nn.BatchNorm1d(CNN_FORCE_conv_BN_layers[index]))

            if self.CNN_FORCE_conv_Dropout_layers != False:
                self.CNN_FORCE_modules.append(nn.Dropout(CNN_FORCE_conv_Dropout_layers[index]))

        for index, i in enumerate(self.CNN_FORCE_linear_layers):
            self.CNN_FORCE_modules.append(nn.Linear(i[0], i[1]))

            if self.CNN_FORCE_linear_activations[index] == "sigmoid":
                self.CNN_FORCE_modules.append(nn.Sigmoid())
            elif self.CNN_FORCE_linear_activations[index] == "relu":
                self.CNN_FORCE_modules.append(nn.ReLU())
            elif self.CNN_FORCE_linear_activations[index] == "tanh":
                self.CNN_FORCE_modules.append(nn.Tanh())
            else:
                self.CNN_FORCE_modules.append(nn.Linear())

            if self.CNN_FORCE_linear_BN_layers[index] != False:
                self.CNN_FORCE_modules.append(nn.BatchNorm1d(CNN_FORCE_linear_BN_layers[index]))

            if self.CNN_FORCE_linear_Dropout_layers != False:
                self.CNN_FORCE_modules.append(nn.Dropout(CNN_FORCE_linear_Dropout_layers[index]))

        self.input_size = C_FORCE_Feature + C_MIC_Feature + C_AE_Feature
        self.hidden_size = 300
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstmcell1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=False,
                                 dropout=self.dropout)

        self.Dense1 = nn.Linear(300, 188)  # the final output should be 17 data points.
        self.Dense1Act = nn.Sigmoid()
        # self.Dense2=nn.Linear(34,17) #It generates 17 data points

    # %%
    # lstm_input_size is determined by the number of features we get from self.features()
    # self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,num_layers=1, batch_first=True)
    # self.classifier = nn.Linear(32, 2)

    # self.optimizer = optim.Adam(self.parameters())
    # self.criterion = nn.NLLLoss()

    def forward(self, AE, Mic, Forces, tillTimeT):

        #  There will be sensor values from 1 to n and m number of features will be produced let`s say for LSTM, Therefore n x m 

        AE_2d_Features = torch.empty(tillTimeT, self.C_AE_Feature)
        Mic_2d_Features = torch.empty(tillTimeT, self.C_MIC_Feature)
        Forces_2d_Features = torch.empty(tillTimeT, self.C_FORCE_Feature)
        LSTMFeature = torch.empty(tillTimeT, )

        for i in range(tillTimeT):
            AE_2d_Features[i, :] = self.CNN_AE_features(AE)  # if T=20 , tensor dim = 20 X C_AE_Feature
            Mic_2d_Features[i, :] = self.CNN_Mic_features(Mic)  # if T=20 , tensor dim = 20 X C_MIC_Feature
            Forces_2d_Features[i, :] = self.CNN_Forces_features(
                Forces)  # if T=20 , tensor dim = 20 X C__Feature
            LSTMFeature[i, :] = torch.cat((AE_2d_Features[i, :], Mic_2d_Features[i, :], Forces_2d_Features[i, :]), 1)

        # LSTMFeature=torch.cat((AE_2d_Features, Mic_2d_Features, Forces_2d_Features), 1)   That might work too.
        print(LSTMFeature.shape)
        LSTMFeature = LSTMFeature.unsqueeze(dim=0)  # adds a 0-th dimension of size 1
        print(LSTMFeature.shape)

        output = self.lstmcell1(LSTMFeature)  # [batch_size, sequence_length, input_dim]
        OutputForceY = self.Dense1Act(self.Dense1(output))

        return OutputForceY


def main():
    # General Structure: Convolutions(Conv->MaxPool->Activation->BN->Dropout) + Linears(LinLayer->Activation->BN->Dropout)
    CNN_AE_conv_layers = [[1, 3, 61, 20],
                          [3, 5, 23, 10]]  # Enter a list for each element [in_channels,out_channels,kernel_size,stride]
    CNN_AE_MaxPoolLayers = [False, []]  # Enter a list for each element [kernel_size,stride] or False if no pool layer
    CNN_AE_conv_layer_activations = ["tanh","relu"]  # Enter "sigmoid","tanh", "linear" or "relu", everything else="linear"
    CNN_AE_conv_BN_layers = [3, False]  # Enter False or num_features for BN, length should be equal to CNN_AE_layers
    CNN_AE_conv_Dropout_layers = [False, 0.1]  # Enter False or propability of dropout
    CNN_AE_linear_layers = [[]]  # Enter tuple (input_size,output_size)
    CNN_AE_linear_activations = []  # Enter "sigmoid","tanh", "linear" or "relu"
    CNN_AE_linear_BN_layers = []
    CNN_AE_linear_Dropout_layers = []

    CNN_AE_properties = [CNN_AE_conv_layers,
                         CNN_AE_MaxPoolLayers,
                         CNN_AE_conv_layer_activations,
                         CNN_AE_conv_BN_layers,
                         CNN_AE_conv_Dropout_layers,
                         CNN_AE_linear_layers,
                         CNN_AE_linear_activations,
                         CNN_AE_linear_BN_layers,
                         CNN_AE_linear_Dropout_layers]  #


if __name__ == "__main__":
    main()
