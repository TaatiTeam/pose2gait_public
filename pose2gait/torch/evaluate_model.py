import logging
import os

import torch
from torch.nn.functional import relu

from pose2gait.evaluation import load_and_evaluate_results

from .checkpoint import load_checkpoint
from .pose2gait_model import Pose2GaitModel

logger = logging.getLogger(__name__)

SOURCE_DICT = {
    "openpose": 0,
    "alphapose": 1,
    "detectron": 2,
}

DATASET_DICT = {
    "ds1": 0,
    "ds2": 1,
}


def evaluate_model(
    run_name,
    fold_name,
    test_dataloader,
    output_dir,
    model_config,
    target_features,
    base_dir=".",
    source_dict=SOURCE_DICT,
    dataset_dict=DATASET_DICT,
):
    """Predict the test set with the best model from a given fold and save the
    results. Then run the evaluation script on the results.

    Args:
        run_name (str): Name of the run to evaluate.
        fold_name (str): Name of the fold to evaluate.
        test_dataloader (torch.utils.data.DataLoader): DataLoader containing
            the test data.
        output_dir (str): Path to the directory where the results should be
            saved.
        model_config (pose2gait.torch.ModelConfig): Configuration of the model.
        target_features (list): List of gait features to predict.
        base_dir (str, optional): Path to the base directory where training
            was run and checkpoints were saved. Defaults to ".".
        source_dict (dict, optional): Dictionary from sequence source names
            to one-hot encoding indices. Defaults to SOURCE_DICT.
        dataset_dict (dict, optional): Dictionary from dataset names
            to one-hot encoding indices. Defaults to DATASET_DICT.
    """
    model = Pose2GaitModel(model_config)
    fold_dir = os.path.join(base_dir, f"runs/{run_name}/{fold_name}")
    model, _, epoch = load_checkpoint("best", model, None, fold_dir)
    logger.info(f"Evaluating {run_name} {fold_name} at epoch {epoch}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results_file = f"{run_name.replace('/', '_')}_{fold_name}_predictions.csv"
    results_file = os.path.join(output_dir, results_file)
    header = []
    for f in target_features:
        header.append(f"pred_{f}")
        header.append(f"gt_{f}")
    datasets = [
        None,
    ] * len(dataset_dict)
    for name, index in dataset_dict.items():
        datasets[index] = name
    header.extend(datasets)
    sources = [
        None,
    ] * len(source_dict)
    for name, index in source_dict.items():
        sources[index] = name
    header.extend(sources)
    with open(results_file, "w") as f:
        f.write(",".join(header))
        f.write("\n")
        for sample, label in test_dataloader:
            locs, meta = sample
            locs = locs.to(device)
            meta = meta.to(device)
            prediction = model((locs, meta))
            prediction = relu(prediction)
            prediction = prediction.detach().cpu().numpy()
            gt = label.numpy()
            meta = meta.detach().cpu().numpy()
            prediction = prediction.squeeze(axis=0)
            gt = gt.squeeze(axis=0)
            meta = meta.squeeze(axis=0)
            pred_str = [str(pred) for pred in prediction]
            gt_str = [str(label) for label in gt]
            meta_str = [str(m) for m in meta]
            line = []
            for p, g in zip(pred_str, gt_str):
                line.append(p)
                line.append(g)
            line.extend(meta_str)
            f.write(",".join(line))
            f.write("\n")
    load_and_evaluate_results(results_file, output_dir, target_features)
