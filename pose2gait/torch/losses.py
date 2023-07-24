import torch


def masked_MSE(input, target):
    # ignore nan label values when computing loss
    mask = torch.isnan(target)
    return torch.nn.MSELoss()(input[~mask], target[~mask])


def masked_MAPE(input, target):
    # divide mae by size of target
    mask = torch.isnan(target)
    masked_input = input[~mask]
    masked_target = target[~mask]
    return (torch.abs((masked_input - masked_target) / masked_target)).mean()


def masked_weighted_MSE(input, target):
    # divide mse by size of target
    mask = torch.isnan(target)
    masked_input = input[~mask]
    masked_target = target[~mask]
    return (((masked_input - masked_target)
             / masked_target.pow(-1)) ** 2).mean()
