import logging

import torch
from torch.nn.functional import relu

logger = logging.getLogger(__name__)


def validate_model(
    model,
    dataloader,
    device,
    criterion,
    feature_weights=None,
):
    """Compute the validation loss for a model.

    Args:
        model (torch.nn.Module): Pytorch model.
        dataloader (torch.utils.data.DataLoader): Validation data.
        device (torch.device): Device to run the model on.
        criterion (torch.loss): Loss function to use for validation.
        feature_weights (list, optional): Multiply predicted and true features
            by these weights before computing loss. Defaults to None.

    Returns:
        float: The validation loss
    """
    model.eval()
    vald_loss = 0.0
    for sample, label in dataloader:
        locs, meta = sample
        locs = locs.to(device)
        meta = meta.to(device)
        prediction = model((locs, meta))
        label = label.to(device)
        if feature_weights:
            weights = torch.tensor(feature_weights).to(device)
            prediction = torch.mul(prediction, weights)
            label = torch.mul(label, weights)
        nonnegative_pred = relu(prediction)
        loss = criterion(nonnegative_pred, label)
        if not torch.isnan(loss):
            vald_loss = vald_loss + loss.item()
    vald_loss = vald_loss / len(dataloader)
    return vald_loss
