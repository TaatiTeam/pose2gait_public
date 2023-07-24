import argparse
import copy
import logging
import os
import random
from dataclasses import dataclass

import toml
from load_datasets import load_dataset
from torch.utils.data import ConcatDataset, DataLoader

from pose2gait.evaluation import combine_folds, run_baseline
from pose2gait.torch import (ModelConfig, Pose2GaitDataset, TrainConfig,
                             evaluate_model, train_model)
from pose2gait.torch.pose2gait_dataset import get_cv_folds_by_person

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
random.seed(34893)

# logging.getLogger("pose2gait.gait_loaders").setLevel(logging.DEBUG)


@dataclass
class DataConfig:
    dataset_names: list
    sources: list
    target_features: list
    normalization: str = "per_video"
    num_frames: int = 120
    mirror: bool = False


def get_folds_from_file(file):
    fold_person_ids = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            fold_person_ids.append(line.split(","))
    return fold_person_ids


def get_folds(datasets, num_folds, base_dir):
    fold_file = os.path.join(base_dir, "fold_person_ids.txt")
    if os.path.isfile(fold_file):
        fold_person_ids = get_folds_from_file(fold_file)
        assert len(fold_person_ids) == num_folds
    else:
        fold_person_ids = [[] for _ in range(num_folds)]
        for ds in datasets:
            dataset_fold_ids = get_cv_folds_by_person(ds, n=num_folds)
            for i, d in enumerate(dataset_fold_ids):
                fold_person_ids[i].extend(d)
        logger.debug(f"Fold ids found {fold_person_ids}")
        with open(fold_file, "w") as f:
            for fold in fold_person_ids:
                f.write(",".join(fold))
                f.write("\n")
    return fold_person_ids


def write_ids_to_file(id_list, filename):
    with open(filename, "w") as f:
        for id in id_list:
            f.write(id + "\n")


def run_experiment(config, datasets):
    # configuration
    num_folds = config["num_folds"]
    run_name = config["run_name"]
    base_dir = config["base_dir"]
    data_cfg = DataConfig(**config["data"])
    dataset_dict = {n: i for i, n in enumerate(data_cfg.dataset_names)}
    source_dict = {n: i for i, n in enumerate(data_cfg.sources)}
    model_config = ModelConfig(
        num_features=len(data_cfg.target_features),
        metadata_len=len(data_cfg.dataset_names) + len(data_cfg.sources),
        input_frames=data_cfg.num_frames,
        **config["model"],
    )
    train_config = TrainConfig(
        run_name=run_name,
        base_dir=base_dir,
        **config["train"],
    )

    # OS setup
    run_dir = os.path.join(base_dir, f"runs/{run_name}")
    results_dir = os.path.join(base_dir, f"results/{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.toml"), "w") as f:
        toml.dump(config, f)

    # set up cross validation folds
    fold_person_ids = get_folds(datasets, num_folds, base_dir)
    for fold_index in range(num_folds):
        fold_name = f"fold{fold_index}"
        fold_dir = os.path.join(run_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        fold_results_dir = os.path.join(results_dir, fold_name)
        os.makedirs(fold_results_dir, exist_ok=True)
        fold_copy = fold_person_ids.copy()
        test_fold = fold_copy.pop(fold_index)
        vald_fold = fold_copy.pop(fold_index - 1)
        train_fold = []
        for f in fold_copy:
            train_fold.extend(f)
        if args.baseline:
            train_fold = train_fold + vald_fold
            vald_fold = []
        write_ids_to_file(train_fold, os.path.join(fold_dir, "train_ids.txt"))
        write_ids_to_file(vald_fold, os.path.join(fold_dir, "vald_ids.txt"))
        write_ids_to_file(test_fold, os.path.join(fold_dir, "test_ids.txt"))
        train_sets = [d.get_subset_by_person(train_fold) for d in datasets]
        vald_sets = [d.get_subset_by_person(vald_fold) for d in datasets]
        test_sets = [d.get_subset_by_person(test_fold) for d in datasets]
        if args.baseline:
            run_baseline(
                train_sets,
                test_sets,
                data_cfg.target_features,
                fold_results_dir,
                dataset_dict,
            )
        else:
            torch_train_set = ConcatDataset(
                [
                    Pose2GaitDataset(
                        d,
                        target_features=data_cfg.target_features,
                        source_dict=source_dict,
                        dataset_dict=dataset_dict,
                    )
                    for d in train_sets
                ]
            )
            logger.debug(f"{len(torch_train_set)} walks in train set")
            torch_vald_set = ConcatDataset(
                [
                    Pose2GaitDataset(
                        d,
                        by_walk=False,
                        target_features=data_cfg.target_features,
                        source_dict=source_dict,
                        dataset_dict=dataset_dict,
                    )
                    for d in vald_sets
                ]
            )
            logger.debug(f"{len(torch_vald_set)} walks in vald set")
            torch_test_set = ConcatDataset(
                [
                    Pose2GaitDataset(
                        d,
                        by_walk=False,
                        target_features=data_cfg.target_features,
                        source_dict=source_dict,
                        dataset_dict=dataset_dict,
                    )
                    for d in test_sets
                ]
            )
            logger.debug(f"{len(torch_test_set)} walks in test set")
            train_dataloader = DataLoader(
                torch_train_set,
                batch_size=train_config.batch_size,
                shuffle=True,
            )
            vald_dataloader = DataLoader(torch_vald_set, shuffle=False)
            test_dataloader = DataLoader(torch_test_set, shuffle=False)

            best_epoch = train_model(
                fold_name,
                train_dataloader,
                vald_dataloader,
                train_config,
                model_config,
            )
            logger.info(f"Best epoch for {run_name}_{fold_name}: {best_epoch}.")
            evaluate_model(
                run_name,
                fold_name,
                test_dataloader,
                fold_results_dir,
                model_config,
                data_cfg.target_features,
                base_dir=base_dir,
                dataset_dict=dataset_dict,
                source_dict=source_dict,
            )
    combine_folds(results_dir, num_folds, data_cfg.target_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("-b", "--baseline", action="store_true")
    parser.add_argument("-c", "--cluster", action="store_true")
    args = parser.parse_args()
    config = toml.load(args.config)

    data_cfg = DataConfig(**config["data"])
    run_name = config["run_name"]
    base_dir = config["base_dir"]

    # setup
    datasets = []
    for dataset in data_cfg.dataset_names:
        datasets.append(
            load_dataset(
                data_cfg.sources,
                dataset,
                cluster=args.cluster,
                seq_length=data_cfg.num_frames,
                norm=data_cfg.normalization,
                mirror=data_cfg.mirror,
            )
        )

    run_experiment(config, datasets)
