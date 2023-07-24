import logging
from dataclasses import dataclass, field

import torch
import torch.nn

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    encoder_channels: list = field(default_factory=lambda: [14, 8])
    kernel_size: int = 3
    linear_features: list = field(default_factory=lambda: [75, 25])

    # these should be inferred from data
    input_frames: int = 120
    num_features: int = 14
    metadata_len: int = 6
    input_joints: int = 12
    input_dims: int = 2


class Pose2GaitModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        """Pytorch model for predicting gait features from pose sequences.
        Includes a shared encoder with 1D convolutions and a MLP predictor
        for each gait feature.

        Args:
            model_config (ModelConfig): The configuration of the model.
        """
        super().__init__()
        self.cfg = model_config
        self.encoder = self.__create_encoder()
        self.predictors = torch.nn.ModuleList(
            [self.__create_predictor() for _ in range(self.cfg.num_features)]
        )

    def __create_encoder(self):
        """Create the shared encoding layers according to the configuration.
        Each layer includes a 1D convolution and a ReLU activation.
        Each entry in self.cfg.encoder_channels represents the output
        channels for one layer.
        Same padding is used to preserve the number of frames.

        Returns:
            torch.nn.Sequential: a sequential set of layers for the encoder
        """
        layers = []
        input_size = self.cfg.input_joints * self.cfg.input_dims
        layer_input_channels = [input_size] + self.cfg.encoder_channels[0:-1]
        layer_output_channels = self.cfg.encoder_channels
        for i, o in zip(layer_input_channels, layer_output_channels):
            layers.append(
                torch.nn.Conv1d(
                    i,
                    o,
                    self.cfg.kernel_size,
                    padding="same",
                )
            )
            layers.append(torch.nn.ReLU())
        return torch.nn.Sequential(*layers)

    def __create_predictor(self):
        """Create one output head for predicting one gait feature.
        Each hidden layer of the predictor includes a linear layer and a ReLU
        activation. self.cfg.linear_features is a list of the output
        dimensions for each hidden layer of the predictor. The final layer is
        a linear layer without a ReLU and has output dimension 1.

        Returns:
            torch.nn.Sequential: a sequential set of layers for the predictor
        """
        layers = []
        channels = self.cfg.encoder_channels[-1]
        frames = self.cfg.input_frames
        additional_info = self.cfg.metadata_len
        input_size = channels * frames + additional_info
        layer_input_dims = [input_size] + self.cfg.linear_features
        layer_output_dims = self.cfg.linear_features + [1]
        for i, o in zip(layer_input_dims, layer_output_dims):
            layers.append(torch.nn.Linear(i, o))
            layers.append(torch.nn.ReLU())
        layers.pop(-1)
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # input has shape (batch_size, t, joints, dims)
        # want (batch_size, joints*dims, t)
        loc_data, metadata = x
        batch_size = loc_data.shape[0]
        loc_data = torch.reshape(
            loc_data,
            (
                batch_size,
                self.cfg.input_frames,
                self.cfg.input_joints * self.cfg.input_dims,
            ),
        )
        loc_data = torch.transpose(loc_data, 2, 1)
        encoded = self.encoder(loc_data)
        flattened = torch.reshape(encoded, (batch_size, -1))
        with_metadata = torch.cat((flattened, metadata), dim=1)
        predicted = [head(with_metadata) for head in self.predictors]
        stacked = torch.stack(predicted, dim=1)
        return stacked.squeeze(dim=-1)
