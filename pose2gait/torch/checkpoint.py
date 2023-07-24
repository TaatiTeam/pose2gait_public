import logging
import os
import shutil

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(epoch, model_state, optim_state, is_best, fold_dir):
    """Save a model checkpoint during training.

    Args:
        epoch (int): Epoch number
        model_state (dict): Output of model.state_dict()
        optim_state (dict): Output of optimizer.state_dict()
        is_best (bool): True if this is the best model so far, and false
            otherwise. If true, saves the checkpoint in a file called
            "best_model.pt" in addition to the usual "epoch_{epoch}.pt."
        fold_dir (str): Path to the directory where the checkpoint should
            be saved.
    """
    state = {
        "epoch": epoch,
        "state_dict": model_state,
        "optimizer": optim_state,
    }
    model_file = os.path.join(fold_dir, f"epoch_{epoch}.pt")
    torch.save(state, model_file)
    if is_best:
        best_file = os.path.join(fold_dir, "best_model.pt")
        shutil.copyfile(model_file, best_file)


def load_checkpoint(epoch, model, optimizer, fold_dir):
    """Load a model checkpoint from disk.

    Args:
        epoch (int, str): Epoch number, or "best" to load the best model.
        model (torch.nn.Module): Model into which to load the saved state.
            Must be of the same dimensions as the model from which the state
            was saved.
        optimizer (torch.optim.Optimizer): The optimizer into which
            to load the saved state. Must be of the type as the one used
            to save the state.
        fold_dir (str): Path to the directory where the checkpoints were
            saved.

    Returns:
        (torch.nn.Module, torch.optim.Optimizer, epoch):
            The model, optimizer, and epoch number loaded from the checkpoint.
    """
    if epoch == "best":
        state_file = os.path.join(fold_dir, "best_model.pt")
    else:
        state_file = os.path.join(fold_dir, f"epoch_{epoch}.pt")
    checkpoint = torch.load(state_file)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]
