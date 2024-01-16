"""This module contains the objects used to process data and create 
model objects.

Note:
`nn.ModuleDict` is required to keep the modules in the models in the 
custom Neural Network model graph.
"""
# --------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------

# Basic libraries
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanAbsolutePercentageError

# Data libraries


# --------------------------------------------------------------------------------------
# CLASSES
# --------------------------------------------------------------------------------------
class TimeCNNRNN(nn.Module):
    def __init__(
        self,
        dense_layers: int,
        dense_units: int,
        dense_batch_norm: bool,
        dense_activation: str,
        rnn_units: int,
        rnn_layers: int,
        rnn_batch_norm: bool,
        rnn_unit_type: str,
        num_vars: int,
        time_steps: int,
        time_window: int,
        cnn_individual_channels: int,
        cnn_synthesized_channels: int,
        cnn_layers: int,
        cnn_batch_norm: bool,
        stride: int = 1,
        n_target_vars: int = 1,
    ) -> nn.Module:
        """
        Initializes the model with the given parameters.

        Args:
            dense_layers (int): The number of dense layers in the model.
            dense_units (int): The number of units in each dense layer.
            dense_batch_norm (bool): Whether to apply batch
                normalization to the dense layers.
            dense_activation (str): The activation function to use for
                the dense layers (except for the last dense layer). It
                should be written as it is called in PyTorch.
            rnn_units (int): The number of units in each RNN layer.
            rnn_layers (int): The number of RNN layers in the model.
            rnn_batch_norm (bool): Whether to apply batch normalization
                to the RNN layers.
            rnn_unit_type (str): The type of RNN unit to use. It should
                be written as it is called in PyTorch.
            num_vars (int): The number of input variables.
            time_steps (int): The number of time steps in the input
                data. It is the time dimension of the input data
                sequence.
            time_window (int): The size of the time window for the
                CNN part of the model (this defines the kernel size
                through time). It must be less than `time_steps`.
            cnn_individual_channels (int): The number of individual
                channels CNN section of the model. This means the
                number of channels to which each of the individual
                variables will be expanded and then convolved on.
            cnn_synthesized_channels (int): The number of synthesized
                channels in the CNN section of the model. This means
                that the CNN synthesize all the variables into this
                number of new variables.
            cnn_layers (int): The number of layers in the CNN layer.
            cnn_batch_norm (bool): Whether to apply batch normalization
                to the CNN layer.
            stride (int, optional): The stride value for the CNN layer.
                Defaults to 1.
            rnn_dropout (float, optional): The dropout rate for the RNN
                layers. Defaults to 0.0.
            n_target_vars (int, optional): The number of target
                variables. Defaults to 1.

        Returns:
            nn.Module: The initialized model.
        """
        # Initiate parent class
        super().__init__()

        # Save attributes
        self.rnn_unit_type = rnn_unit_type

        #  -------------------------- Define layers ------------------------------------

        # Time Convolutional Network
        self.time_cnn = TimeCNN(
            num_vars=num_vars,
            time_steps=time_steps,
            time_window=time_window,
            cnn_individual_channels=cnn_individual_channels,
            cnn_synthesized_channels=cnn_synthesized_channels,
            num_layers=cnn_layers,
            batch_norm=cnn_batch_norm,
            stride=stride,
        )

        # Time RNN
        rnn_layer_list = list()

        for i in range(rnn_layers):
            # Define cycle variables
            if i == 0:
                rnn_vars = num_vars * 2 + cnn_synthesized_channels
            else:
                rnn_vars = rnn_units

            # Add batch normalization layer
            if rnn_batch_norm:
                rnn_layer_list.append(
                    (f"rnn_batch_norm_{i + 1}", nn.BatchNorm1d(rnn_vars))
                )

            # Add RNN layer
            rnn_layer_list.append(
                (
                    f"rnn_{rnn_unit_type}_{i + 1}",
                    getattr(nn, rnn_unit_type)(
                        input_size=rnn_vars,
                        hidden_size=rnn_units,
                        batch_first=True,
                    ),
                )
            )

        self.rnn = nn.ModuleDict(rnn_layer_list)

        # Add dense layer after RNN layers
        dense_layer_list = list()

        for i in range(dense_layers):
            # Define cycle variables
            if i == dense_layers - 1:
                dense_n_units = n_target_vars
            else:
                dense_n_units = dense_units

            if i == 0:
                input_dense_units = rnn_units
            else:
                input_dense_units = dense_units

            # Add batch normalization layer
            if dense_batch_norm:
                dense_layer_list.append(
                    (f"dense_batch_norm_{i + 1}", nn.BatchNorm1d(input_dense_units))
                )

            # Add dense layer
            dense_layer_list.append(
                (
                    f"dense_{i + 1}",
                    nn.Linear(input_dense_units, dense_n_units),
                )
            )

            # Add activation
            if i < dense_layers - 1:
                dense_layer_list.append(
                    (
                        f"dense_activation_{dense_activation}_{i + 1}",
                        getattr(nn, dense_activation)(),
                    )
                )

        self.dense = nn.Sequential(OrderedDict(dense_layer_list))

        # Initialize weights
        self.initialize_rnn_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through the time convolutional network: feature engineering
        cnn_x = self.time_cnn(x)

        # Concatenate time CNN features with original data
        output = torch.cat([x, cnn_x], dim=2)

        # Pass through the RNN network
        for name, layer in self.rnn.items():
            if "batch_norm" not in name:
                output, _ = layer(output)

            else:
                # BatchNorm1d receives the inputs as (batch, variables, time)
                output = layer(output.permute(0, 2, 1)).permute(0, 2, 1)

        # Select only last output from the RNN sequence
        output = output[:, -1, :]

        # Pass through the dense network
        output = self.dense(output)

        return output

    def initialize_rnn_weights(self) -> None:
        # Compute the gain
        gain = nn.init.calculate_gain("tanh")

        # Initialize the weights for all LSTM layer modules
        for m in self.modules():
            # If RNN unit is LSTM or GRU
            if isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                nn.init.xavier_uniform_(m.weight_hh_l0, gain=gain)
                nn.init.xavier_uniform_(m.weight_ih_l0, gain=gain)


class TimeCNN(nn.Module):
    def __init__(
        self,
        num_vars: int,
        time_steps: int,
        time_window: int,
        cnn_individual_channels: int,
        cnn_synthesized_channels: int,
        num_layers: int,
        batch_norm: bool,
        stride: int = 1,
    ):
        """
        Initializes the object with the given parameters.

        Args:
            num_vars (int): The number of variables.
            time_steps (int): The number of time steps.
            time_window (int): The size of the time window for
                convolution.
            cnn_individual_channels (int): The number of channels for
                individual variable CNN.
            cnn_synthesized_channels (int): The number of channels for
                synthesized variable CNN. It is the number of
                synthesized variables that will be concatenated with the
                individual variables.
            num_layers (int): The number of layers.
            batch_norm (bool): A flag indicating whether to use batch
                normalization.
            stride (int, optional): The stride for convolution. Defaults
                to 1.

        Returns:
            None
        """
        # Initiate parent class
        super().__init__()

        # Save important values
        self.num_vars = num_vars
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        #  -------------------------- Define layers ------------------------------------

        """
        + `individual_var_layers` list contain convolutions objects that go only
        through time, meaning an individual variable treatment.
        
        + `synthesize_var_layers` contain convolution objects that go both through time
        and through all the variables at the same time; this results in only one 
        variable through time as output, where you can join multiple convolutions like
        this through time as a new set of features.
        """
        individual_var_layers = list()
        synthesized_var_layers = list()
        same_padding = compute_dim_same_padding(n=time_steps, k=time_window, s=stride)

        if batch_norm:
            individual_batch_norm_list = list()
            synthesized_batch_norm_list = list()

        for i in range(num_layers):
            # Define condition for first layer
            if i == 0:
                # Define the number of synthesized variables
                synthesized_nvars = num_vars

            # Define condition for other layers
            else:
                # Define the number of synthesized variables
                synthesized_nvars = cnn_synthesized_channels

            # Fill the `batch_norm_list` with tuples
            if batch_norm:
                individual_batch_norm_list.append(
                    (
                        f"individual_batch_norm_{i+1}",
                        nn.BatchNorm1d(num_vars),
                    )
                )
                synthesized_batch_norm_list.append(
                    (
                        f"synthesized_batch_norm_{i+1}",
                        nn.BatchNorm1d(synthesized_nvars),
                    )
                )

            # Fill the `individual_var_layers`
            individual_var_layers.append(
                (
                    f"individual_conv_{i+1}",
                    nn.Sequential(
                        OrderedDict(
                            [
                                # Individual variable convolution
                                (
                                    f"individual_conv_open_{i+1}",
                                    nn.Conv2d(
                                        in_channels=1,
                                        out_channels=cnn_individual_channels,
                                        kernel_size=(time_window, 1),
                                        padding=(same_padding, 0),
                                    ),
                                ),
                                (f"individual_tanh_open_{i+1}", nn.Tanh()),
                                # Collapse the additional channels
                                (
                                    f"individual_conv_close_{i+1}",
                                    nn.Conv2d(
                                        in_channels=cnn_individual_channels,
                                        out_channels=1,
                                        kernel_size=(time_window, 1),
                                        padding=(same_padding, 0),
                                    ),
                                ),
                                (f"individual_tanh_close_{i+1}", nn.Tanh()),
                            ]
                        )
                    ),
                )
            )

            # Fill the `synthesized_var_layers`
            synthesized_var_layers.append(
                (
                    f"synthesized_conv_{i+1}",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    f"synthesized_conv_{i+1}",
                                    nn.Conv2d(
                                        in_channels=1,
                                        out_channels=cnn_synthesized_channels,
                                        kernel_size=(time_window, synthesized_nvars),
                                        padding=(same_padding, 0),
                                    ),
                                ),
                                (f"synthesized_tanh_{i+1}", nn.Tanh()),
                            ]
                        )
                    ),
                )
            )

        # Save into ModuleDict the different layers
        if batch_norm:
            self.individual_batch_norm_layers = nn.ModuleDict(
                individual_batch_norm_list
            )
            self.synthesized_batch_norm_layers = nn.ModuleDict(
                synthesized_batch_norm_list
            )
        self.synthesized_var_layers = nn.ModuleDict(synthesized_var_layers)
        self.individual_var_layers = nn.ModuleDict(individual_var_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network. It is composed of two
        sections:

            - Individual convolution: convolves through time by
                each individual variable.
            - Synthesized convolution: convolves through time using all
                the variables at the same time. Based on the number of
                synthesized channels, generates new variables.

        After reshape the outputs of the parallel convolutions, it joins
        them through the variable dimension.

        Parameters:
            x (torch.Tensor): Input tensor of shape
                (batch_size, time_steps, num_vars).

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, time_steps, 2*num_vars).
        """
        # Assert dimensionality of input
        assert (x.shape[1] == self.time_steps) and (x.shape[2] == self.num_vars), (
            f"Wrong dimensionality of input: {x.shape} | Expected: "
            f"(batch, {self.time_steps}, {self.num_vars})"
        )

        # Generate inputs for both sides (individual and synthesized) convolutions
        individual_x = x
        synthesized_x = x

        # Forward pass through layers
        for i in range(self.num_layers):
            # Adjust Batch Normalization if True
            if self.batch_norm:
                # BatchNorm1d receives the inputs as (batch, variables, time)
                individual_x = list(self.individual_batch_norm_layers.values())[i](
                    individual_x.permute(0, 2, 1)
                ).permute(0, 2, 1)
                synthesized_x = list(self.synthesized_batch_norm_layers.values())[i](
                    synthesized_x.permute(0, 2, 1)
                ).permute(0, 2, 1)

            # Expand channel dimension
            individual_x = individual_x.unsqueeze(1)
            synthesized_x = synthesized_x.unsqueeze(1)

            # Individual Variable Convolution
            individual_x = list(self.individual_var_layers.values())[i](
                individual_x
            ).squeeze(1)

            # Synthesized Variable Convolution
            synthesized_x = (
                list(self.synthesized_var_layers.values())[i](synthesized_x)
                .permute(0, 3, 2, 1)  # convert synthesized channels into variables
                .squeeze(1)  # remove channel dimension (1 after permutation)
            )

        # Concatenate through the variable dimension and return output
        output = torch.cat([individual_x, synthesized_x], dim=2)

        return output


class EarlyStopper(object):
    def __init__(self, patience: int = 10, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False

    def __call__(self, validation_loss: float) -> None:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# --------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------
def compute_dim_same_padding(n: int, k: int, s: int) -> int:
    """
    Compute the padding based on the given parameters.

    Args:
        n (int): dimension of input.
        k (int): Kernel/filter dimension size.
        s (int): The stride.

    Returns:
        int: The computed padding.
    """
    return int(((s - 1) * n - s + k) / 2)


def series_to_sequence(
    data: pd.DataFrame,
    num_target_vars: int,
    window_size: int = 6,
    test_split: float = 0.3,
) -> tuple:
    """This function transforms series of historical or sequential
    data into an ordered sequence of sequences. The output contains a
    sequence of sequences of size equal to window_size. The input data
    is also divided into train and test datasets so that the last
    train_test_split proportion of data is part of the test dataset.

    Args:
        data (np.ndarray): array-like
            data structure with the series of reshaped data.
        window_size (int, optional): size of the sequence window.
            Defaults to 6.
        test_split (float, optional): proportion of the tail of the
            input data that is assigned as the test dataset. Defaults to
            0.3 (30%).

    Returns:
        tuple: tuple of arrays (x_train, y_train, x_test, y_test)

            - x_train: np.ndarray with the predictive variables. Its
                shape is
                ((data.shape[0] - window_size) * (1 - test_split),
                # of variables).

            - y_train: np.array with the target variables. Its shape is
                ((data.shape[0] - window_size) * (1 - test_split),
                # of variables).

            - x_test: np.ndarray with the test predictive variables.

            - y_test: np.ndarray with the test target variables.
    """
    data = data.values
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    # Create the sequence of window_size sized sequences:
    data_t = []
    for index in range(len(data) - window_size + 1):
        data_t.append(data[index : index + window_size])

    # Convert list of sequences into array:
    data_t = np.array(data_t)

    # Train-test split:
    split = int(data_t.shape[0] * (1 - test_split))
    x_train = data_t[:split, :, :-num_target_vars]
    y_train = data_t[:split, -1, -num_target_vars:]
    x_test = data_t[split:, :, :-num_target_vars]
    y_test = data_t[split:, -1, -num_target_vars:]

    return (x_train, y_train, x_test, y_test)


def train_RNNCNN_model(
    data: pd.DataFrame,
    train_test_split: float,
    cnn_time_window: int,
    device: torch.device,
    cnn_individual_channels: int,
    cnn_synthesized_channels: int,
    cnn_layers: int,
    cnn_batch_norm: bool,
    rnn_units: int,
    rnn_layers: int,
    rnn_batch_norm: bool,
    rnn_unit_type: str,
    dense_layers: int,
    dense_units: int,
    dense_batch_norm: bool,
    dense_activation: str,
    time_steps: int,
    batch_size: int,
    num_target_vars: int,
    epochs: int,
    learning_rate: float,
    optimizer_name: str,
    patience: int,
    model_training_tries: int = 1,
    save_best_model_path: str = None,
) -> TimeCNNRNN:
    # ----------------------------  Build Training-Test Datasets -----------------------
    # Build batch x time_steps x variables arrays
    x_train, y_train, x_test, y_test = series_to_sequence(
        data=data,
        window_size=time_steps,
        num_target_vars=num_target_vars,
        test_split=train_test_split,
    )

    # Build the Tensor Datasets
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.float),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
    )

    # Build PyTorch data loaders
    if len(train_dataset) % batch_size == 1:
        drop_last = True
    else:
        drop_last = False

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    if len(val_dataset) % batch_size == 1:
        drop_last = True
    else:
        drop_last = False

    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    # --------------------------------  Build Model ------------------------------------
    # Setup the tries to train the model cycle
    best_mape = np.inf

    for training_try in range(model_training_tries):
        # Best model mark
        is_best_model = False

        # Neural network model
        model = TimeCNNRNN(
            dense_layers=dense_layers,
            dense_units=dense_units,
            dense_batch_norm=dense_batch_norm,
            dense_activation=dense_activation,
            rnn_units=rnn_units,
            rnn_layers=rnn_layers,
            rnn_batch_norm=rnn_batch_norm,
            rnn_unit_type=rnn_unit_type,
            num_vars=x_train.shape[-1],
            time_steps=time_steps,
            time_window=cnn_time_window,
            cnn_individual_channels=cnn_individual_channels,
            cnn_synthesized_channels=cnn_synthesized_channels,
            cnn_layers=cnn_layers,
            cnn_batch_norm=cnn_batch_norm,
            n_target_vars=y_train.shape[-1],
        ).to(device)

        # Define optimizer
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

        # Define learning rate scheduler:
        """This scheduler looks to the loss: if it hasn't decreased for `patience` 
        number of epochs, then it reduces the learning rate. You can find the
        documentation here:
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )

        # Define the loss function
        criterion = nn.MSELoss()

        # Define validation function: send to device as well
        mape_val_function = MeanAbsolutePercentageError().to(device)

        # Define data frame with epoch results
        results_df = pd.DataFrame(
            columns=["epoch", "train_loss", "val_loss", "val_mape"]
        )

        # Define early stopper
        early_stopper = EarlyStopper(patience=patience, min_delta=0)

        # --------------------------------  Training Loop ------------------------------
        for epoch in range(epochs):
            # Save batch losses for the scheduler
            epoch_losses = list()

            # Set the model to train mode
            model.train()

            # Iterate through batches
            for x, y in train_dataloader:
                # Send tensors to device (GPU if available)
                x, y = x.to(device), y.to(device)

                # Reset gradients for the optimizer
                optimizer.zero_grad()

                # Compute model output
                output = model(x)

                # Compute loss and back-propagate loss gradients
                loss = criterion(output, y)
                epoch_losses.append(loss.item())
                loss.backward()

                # Optimization step
                optimizer.step()

            # Compute the epoch average loss value
            mean_loss = sum(epoch_losses) / len(epoch_losses)

            # Give a step with scheduler
            scheduler.step(mean_loss)

            # ----------------------------  Validation Loop ----------------------------
            # Set the model to evaluation mode
            model.eval()

            # Set the loop with no gradient
            with torch.no_grad():
                mape_list = list()
                val_loss_list = list()

                # Iterate through batches
                for x_val, y_val in val_dataloader:
                    # Send tensors to device: GPU if available
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    # Compute output from model
                    output = model(x_val)

                    # Compute MAPE and loss functions
                    batch_mape = mape_val_function(output, y_val)
                    batch_loss = criterion(output, y_val)
                    mape_list.append(batch_mape.item())
                    val_loss_list.append(batch_loss.item())

            val_mape = sum(mape_list) / len(mape_list)
            val_loss = sum(val_loss_list) / len(val_loss_list)

            # Save results
            results_df.loc[epoch] = [epoch, mean_loss, val_loss, val_mape]

            # Save best model `state_dict`
            if val_mape < best_mape and save_best_model_path:
                # Update best values
                best_model = model
                is_best_model = True
                best_mape = val_mape

                # Save best model
                torch.save(model.state_dict(), save_best_model_path)

            # Early stopping
            early_stopper(val_mape)

            if early_stopper.stop:
                print("Early stopping: Epoch {}".format(epoch))
                break

        # Save best model metadata
        if is_best_model:
            best_results_df = results_df.copy()
            print(f"Best model found at try {training_try} with MAPE {best_mape}\n")

    return best_model, best_results_df


def regression_report(y_observed: np.ndarray, y_predicted: np.ndarray) -> pd.DataFrame:
    """
    Calculate regression metrics based on the observed and predicted
    values.

    Args:
        y_observed (np.ndarray): The observed values of the target
            variable.
        y_predicted (np.ndarray): The predicted values of the target
            variable.

    Returns:
        pd.DataFrame: A dataframe containing the calculated regression
            metrics.The metrics include R2, RMSE, MAE, and MAPE.
    """
    # Create metrics data frame
    df = pd.DataFrame(index=["R2", "RMSE", "MAE", "MAPE"], columns=["Value"])

    # Compute metrics
    df.loc["R2", "Value"] = r2_score(y_observed, y_predicted)
    df.loc["RMSE", "Value"] = mean_squared_error(y_observed, y_predicted, squared=False)
    df.loc["MAE", "Value"] = mean_absolute_error(y_observed, y_predicted)
    df.loc["MAPE", "Value"] = mean_absolute_percentage_error(y_observed, y_predicted)

    return df


def generate_forecast(
    model: nn.Module,
    df: pd.DataFrame,
    n_steps: int,
    time_window: int,
    device: str,
    time_unit: str = "D",
) -> pd.DataFrame:
    """
    Generates a forecast using a given model.

    Args:
        model (nn.Module): The model to use for generating the forecast.
        df (pd.DataFrame): The input data frame. Must contain all the
            predictive variables, and the target variable in the last
            position.
        n_steps (int): The number of steps to forecast.
        time_window (int): The window size to use for the modeling.
        device (str): The device to use for computation.
        time_unit (str, optional): The time unit for forecasting.
            Defaults to "D", which means daily.

    Returns:
        pd.DataFrame: The data frame containing the forecast.
    """

    # Set the model to eval mode
    model.eval()

    # Generate a copy of the input data frame to avoid mutating the original data
    output_df = df.copy()

    # Cycle through the number of steps
    for step in range(1, n_steps + 1):
        # Compute the next date of the forecast
        next_date = output_df.index.max() + np.timedelta64(1, time_unit)

        # Prepare the data: reshape and then convert to tensor
        x, _, _, _ = series_to_sequence(
            data=output_df.iloc[time_window:, :],
            window_size=time_window,
            num_target_vars=1,
            test_split=0.0,
        )
        x = torch.tensor(x, dtype=torch.float).to(device)

        # Compute the output
        with torch.no_grad():
            y = model(x).to("cpu").numpy()[0]

            # Store the model's output in the output data frame
            output_df.loc[next_date, output_df.columns[-1]] = y

    return output_df
