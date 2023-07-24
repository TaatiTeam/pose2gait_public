import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .losses import masked_MAPE, masked_MSE, masked_weighted_MSE
from .pose2gait_model import ModelConfig, Pose2GaitModel
from .validate_model import validate_model

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    run_name: str
    loss_function: str
    base_dir: str = "."
    epochs: int = 100
    batch_size: int = 20
    save_every: int = 10
    lr: float = 1e-5
    start_epoch: int = 1
    feature_weights: list = None


def write_stats_to_log(log_file, epoch, train_loss, vald_loss, time):
    with open(log_file, "a") as f:
        f.write(f"{epoch},{train_loss},{vald_loss},{time}\n")


def train_model(
    fold_name,
    dataloader,
    vald_loader,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
):
    """Run training for a model including validation after each epoch.

    Args:
        fold_name (str): Name of the fold that is to be trained.
        dataloader (torch.utils.data.DataLoader): Training data.
        vald_loader (torch.utils.data.DataLoader): Validation data.
        train_cfg (TrainConfig): Training hyperparameters.
        model_cfg (ModelConfig): Model hyperparameters.

    Raises:
        ValueError: if the pose_sequences do not specify the source in their
            seq_id.
        ValueError: If the pose_sequences do not specify the dataset in their
            metadata.

    Returns:
        int: The epoch with the lowest validation loss.
    """
    # filesystem setup
    run_dir = os.path.join(train_cfg.base_dir, f"runs/{train_cfg.run_name}")
    fold_dir = os.path.join(run_dir, fold_name)
    log_file = os.path.join(fold_dir, "train_log.txt")

    os.makedirs(fold_dir, exist_ok=True)
    vald_loss_log = os.path.join(fold_dir, "validation_loss.txt")
    # model setup
    model = Pose2GaitModel(model_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_name = train_cfg.loss_function
    if loss_name == "MSE":
        loss_function = masked_MSE
    elif loss_name == "MAPE":
        loss_function = masked_MAPE
    elif loss_name == "weightedMSE":
        loss_function = masked_weighted_MSE
    else:
        raise ValueError(f"Requested loss {loss_name} is not supported.")
    min_best_epoch = train_cfg.epochs // 10

    # load saved state
    if train_cfg.start_epoch > 1:
        with open(vald_loss_log) as f:
            lines = f.readlines()
            line = lines[-1]
            epoch, loss = line.strip().split(",")
            min_validation_loss = float(loss)
            min_validation_epoch = int(epoch)
        model, optim, _ = load_checkpoint(
            train_cfg.start_epoch,
            model,
            optim,
            fold_dir,
        )
    else:
        min_validation_loss = np.inf
        min_validation_epoch = None
        if os.path.isfile(vald_loss_log):
            os.remove(vald_loss_log)
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,vald_loss,time\n")

    for epoch in range(train_cfg.start_epoch, train_cfg.epochs + 1):
        start_time = time.time()
        model.train()
        epoch_loss = []
        for sample, label in dataloader:
            joint_locs, metadata = sample
            if torch.any(torch.isnan(joint_locs)):
                raise ValueError("Found NAN in input!!")
            joint_locs = joint_locs.to(device)
            metadata = metadata.to(device)
            prediction = model((joint_locs, metadata))
            logger.debug(f"predicted: {prediction.detach().cpu().numpy()}")
            logger.debug(f"label: {label}")
            label = label.to(device)
            if train_cfg.feature_weights is not None:
                assert len(train_cfg.feature_weights) == model_cfg.num_features
                weights = torch.tensor(train_cfg.feature_weights).to(device)
                prediction = torch.mul(prediction, weights)
                label = torch.mul(label, weights)
            loss = loss_function(prediction, label)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optim.step()
            epoch_loss.append(loss.detach().cpu().numpy())
        mean_epoch_loss = np.mean(epoch_loss)
        epoch_time = time.time() - start_time
        vald_loss = validate_model(model, vald_loader, device, loss_function)
        write_stats_to_log(
            log_file,
            epoch,
            mean_epoch_loss,
            vald_loss,
            epoch_time,
        )
        save = False
        best = False
        if vald_loss < min_validation_loss and epoch > min_best_epoch:
            logger.info(f"Validation loss decreased to {vald_loss} at epoch {epoch}")
            min_validation_loss = vald_loss
            min_validation_epoch = epoch
            save = True
            best = True
        if epoch % train_cfg.save_every == 0:
            save = True
        if save:
            model_state = model.state_dict()
            optimizer_state = optim.state_dict()
            save_checkpoint(
                epoch,
                model_state,
                optimizer_state,
                best,
                fold_dir,
            )
            with open(vald_loss_log, "a") as f:
                f.write(f"{epoch},{vald_loss}")
                f.write("\n")
    return min_validation_epoch
