"""
Training methods for `LSTMModel`.

@author(2021_05_10): Connor W. Colombo (colombo@cmu.edu)
"""
from typing import Optional, List, Any, Dict, Tuple, Union
import attr

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore

from model import LSTMModel
from data import MachiningDataset
from logger import logger
from opts import get_opts

opts = get_opts()


@attr.s(frozen=True, cmp=True, slots=True, auto_attribs=True)
class TrainHyperparams:
    """Collection of tunable hyperparameters for the training process."""
    # Number of training epochs:
    num_epochs: int
    # Adam learning rate:
    lr: float
    # Batch size:
    batch_size: int
    # Proportion of total dataset which should be used for validation:
    validation_split: float
    # Proportion of total dataset which should be used for final testing:
    test_split: float
    # Which channel number to start forecasting at:
    channel_forecast_start: int
    # How many channels to step by when increasing forecast distance:
    channel_forecast_step: int


def train(
    model: LSTMModel,
    dataset: MachiningDataset,
    random_seed: int = 42,
    hyperparams: Optional[TrainHyperparams] = None
) -> LSTMModel:
    """
    Trains the given `LSTMModel` `model` on the given `data` and returns the trained model.
    """
    sns.set_theme()

    # check if cuda available and use it:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info( # type: ignore
        f"Using device {device}."
    )
    model.to(device)

    if hyperparams is None:
        hyperparams = TrainHyperparams(
            num_epochs=opts.num_epochs,
            lr=opts.lr,
            batch_size=opts.batch_size,
            validation_split=opts.validation_split,
            test_split=opts.test_split,
            channel_forecast_start=opts.channel_forecast_start,
            channel_forecast_step=opts.channel_forecast_step
        )

    dataset_size = len(dataset)

    if hyperparams.batch_size > dataset_size:
        raise ValueError(
            f"Batch size ({hyperparams.batch_size}) can't be greater than "
            f"total number of examples in dataset ({dataset_size})."
        )

    if (hyperparams.validation_split + hyperparams.test_split) >= 1.0:
        raise ValueError(
            f"Validation split ({hyperparams.validation_split*100:3.1f}%) "
            f"and test split ({hyperparams.test_split*100:3.1f}%) "
            f"can't add up to >= 100% since there has to be samples left over "
            f"for training (ideally >50% for training)."
        )
    if (hyperparams.validation_split + hyperparams.test_split) > 0.5:
        logger.warning(
            f"Validation split ({hyperparams.validation_split*100:3.1f}%) "
            f"and test split ({hyperparams.test_split*100:3.1f}%) "
            f"currently add up to {(hyperparams.validation_split + hyperparams.test_split)*100:3.1f}%. "
            f"Ideally, they should add up to less than 50% so most of the data will be used for training. "
        )

    # Creating data indices for training and validation splits:
    indices = [*range(dataset_size)]

    splits = [
        int(hyperparams.validation_split * dataset_size),
        int((hyperparams.validation_split+hyperparams.test_split) * dataset_size)
    ]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    val_indices = indices[:splits[0]]
    test_indices = indices[splits[0]:splits[1]]
    train_indices = indices[splits[1]:]

    if len(val_indices) < 2 or len(test_indices) < 2 or len(train_indices) < 2:
        raise ValueError(
            "Dataset is partitioned poorly. Testing, Validation, and Training "
            "should each have at least two examples (ideally a lot more) but "
            f"instead dataset contains {len(dataset)} `DataExample`s split into: "
            f"{(1-hyperparams.validation_split-hyperparams.test_split)*100:3.1f}% = {len(train_indices)} for training, "
            f"{hyperparams.validation_split*100:3.1f}% = {len(val_indices)} for validation, "
            f"and {hyperparams.test_split*100:3.1f}% = {len(test_indices)} for final testing."
        )

    logger.verbose(  # type: ignore
        f"Dataset contains {len(dataset)} `DataExample`s split into: "
        f"{(1-hyperparams.validation_split-hyperparams.test_split)*100:3.1f}% = {len(train_indices)} for training, "
        f"{hyperparams.validation_split*100:3.1f}% = {len(val_indices)} for validation, "
        f"and {hyperparams.test_split*100:3.1f}% = {len(test_indices)} for final testing."
    )

    # Creating PT data samplers and loaders:
    train_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=SubsetRandomSampler(train_indices)
    )
    val_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=SubsetRandomSampler(val_indices)
    )
    test_loader = DataLoader(
        dataset,
        batch_size=hyperparams.batch_size,
        sampler=SubsetRandomSampler(test_indices)
    )

    # Setup:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.lr)

    logger.info("Beginning training . . .")

    train_loss_history: List[float] = []
    validation_loss_history: List[float] = []
    for epoch in range(hyperparams.num_epochs):
        epoch_start_time = time.time()  # record run TIME

        # Run over training data:
        model.train()
        train_loss_history.append(0)
        num_inst: int = 0 # number of instances in epoch
        for batch_idx, ex in enumerate(train_loader):
            stack = ex['stack']
            num_in_batch = len(ex['spindle_speed'])
            for j in range(num_in_batch):
                total_num_channels = stack['signalAE'][j].shape[0]
                channel_nums = [*range(total_num_channels)]

                start = hyperparams.channel_forecast_start
                step = hyperparams.channel_forecast_step
                stop = total_num_channels - step  # leave one more step to forecast

                for tillChannelN in channel_nums[start:stop:step]:
                    optimizer.zero_grad()
                    outputs = model.forward(
                        stack['signalAE'][j].to(device), stack['signalMic'][j].to(device), stack['signalForces'][j].to(device),
                        tillChannelN,
                        device=device
                    ).to(device)
                    # OutputForceY.size() torch.Size([1, 125])
                    loss = criterion(
                        outputs.double().flatten(),
                        stack['signalFy'][j][tillChannelN + step, :].to(device)
                    ).to(device)
                    loss.backward()
                    optimizer.step()

                    train_loss_history[-1] += loss.item() # sum up across all instances in epoch
                    num_inst += 1
        train_loss_history[-1] = train_loss_history[-1] / num_inst # average across all instances in epoch

        # Compare performance on validation set:
        model.eval()
        validation_loss_history.append(0)
        num_inst: int = 0 # number of instances in epoch
        for batch_idx, ex in enumerate(val_loader):
            stack = ex['stack']
            num_in_batch = len(ex['spindle_speed'])
            for j in range(num_in_batch):
                total_num_channels = stack['signalAE'][j].shape[0]
                channel_nums = [*range(total_num_channels)]

                start = hyperparams.channel_forecast_start
                step = hyperparams.channel_forecast_step
                stop = total_num_channels - step  # leave one more step to forecast

                for tillChannelN in channel_nums[start:stop:step]:
                    outputs = model.forward(
                        stack['signalAE'][j].to(device), stack['signalMic'][j].to(device), stack['signalForces'][j].to(device),
                        tillChannelN,
                        device=device
                    ).to(device)
                    # OutputForceY.size() torch.Size([1, 125])
                    loss = criterion(
                        outputs.double().flatten(),
                        stack['signalFy'][j][tillChannelN + step, :].to(device)
                    ).to(device)

                    validation_loss_history[-1] += loss.item() # sum up across all instances in epoch
                    num_inst += 1
        validation_loss_history[-1] = validation_loss_history[-1] / num_inst # average across all instances in epoch

        epoch_end_time = time.time()
        logger.verbose(  # type: ignore
            f"Epoch [{epoch}/{hyperparams.num_epochs-1}]: {(epoch_end_time-epoch_start_time):5.1f}s \t "
            f"— Mean Training Loss: {train_loss_history[-1]} \t"
            f"— Mean Validation Loss: {validation_loss_history[-1]} \t"
        )

    
    # Collect performance of trained model on test set:
    model.eval()
    test_mean_mse = 0
    num_inst: int = 0 # number of instances in test set
    for batch_idx, ex in enumerate(test_loader):
        stack = ex['stack']
        num_in_batch = len(ex['spindle_speed'])
        for j in range(num_in_batch):
            total_num_channels = stack['signalAE'][j].shape[0]
            channel_nums = [*range(total_num_channels)]

            start = hyperparams.channel_forecast_start
            step = hyperparams.channel_forecast_step
            stop = total_num_channels - step  # leave one more step to forecast

            for tillChannelN in channel_nums[start:stop:step]:
                outputs = model.forward(
                    stack['signalAE'][j].to(device), stack['signalMic'][j].to(device), stack['signalForces'][j].to(device),
                    tillChannelN,
                    device=device
                ).to(device)
                # OutputForceY.size() torch.Size([1, 125])
                loss = criterion(
                    outputs.double().flatten(),
                    stack['signalFy'][j][tillChannelN + step, :].to(device)
                ).to(device)

                test_mean_mse += loss.item() # sum up across all instances in epoch
                num_inst += 1
    test_mean_mse = test_mean_mse / num_inst # average across all instances in epoch

    logger.success(  # type: ignore
        f"Training complete. "
        f"Mean MSE on Test Set: {test_mean_mse} "
    )

    # Plot performance over training:
    plt.figure()
    plt.plot(train_loss_history)
    plt.plot(validation_loss_history)
    plt.xlabel('Epoch Number')
    plt.ylabel('Mean MSE Loss in Epoch')
    plt.legend(('Training Set', 'Validation Set'))
    plt.show()

    # Plot Results for end of test set:
    plt.figure()

    y_pred = outputs.squeeze(dim=0).detach().cpu().numpy()
    y_real = stack['signalFy'][j][tillChannelN+step, :].detach().cpu().numpy()
    plt.plot(np.linspace(0, 360, y_pred.size), y_pred)
    plt.plot(np.linspace(0, 360, y_real.size), y_real)

    plt.xlabel('Bit Rotation Angle [deg]')
    plt.ylabel('Normalized Force')
    plt.legend(('Prediction', 'Real'))
    plt.title(f'LSTM Force Prediction for Channel {tillChannelN+step}')
    plt.show()

    return model
