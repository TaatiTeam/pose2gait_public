import logging
import os

from pose2gait import Dataset
from pose2gait.gait_loaders import KinectLoader, DS2KinectLoader
from pose2gait.sequence_loaders import (CSVLoader, get_ds1_metadata,
                                        get_ds2_metadata)

logger = logging.getLogger(__name__)

LOCAL_BASE_PATH = "*****"
CLUSTER_BASE_PATH = "*******"


def load_sequences(sources, path, metadata_func, person_ids=None):
    """Generic function for loading pose sequences from any dataset

    Args:
        sources (list): list of subdirectories within path to load pose
            sequences from. These generally represent the different sources of
            pose sequences, e.g. (openpose, alphapose, detectron), but each
            dataset names the subdirectories differently.
        path (str): base path to load pose sequences from
        metadata_func (Callable): function to get metadata for the particular
            dataset
        person_ids (list, optional): List of person_ids to include in the
            returned sequences. Used for getting cross validation folds.
            Defaults to None, which includes all sequences.

    Returns:
        list of pose_sequence.PoseSequence: a list of pose sequences loaded
            from the specified locations.
    """
    seqs = []
    for source in sources:
        loader = CSVLoader(os.path.join(path, source), metadata_func)
        for seq in loader.load_sequences(lazy=True):
            if person_ids is None or seq.metadata["person_id"] in person_ids:
                seqs.append(seq)
        logger.info(f"{source} loaded")
    return seqs


def load_dataset(
    sources,
    dataset,
    person_ids=None,
    cluster=False,
    seq_length=120,
    norm="per_video",
    mirror=False,
):
    logger.info(f"Loading {dataset}")
    if dataset == "ds1":
        ds_info = ds1_dataset_info(sources, cluster=cluster)
    elif dataset == "ds2":
        ds_info = ds2_dataset_info(sources, cluster=cluster)
    else:
        raise ValueError(f"Provided dataset {dataset} cannot be loaded")

    seq_path, sources, meta_func, gait_loader, feats_path = ds_info
    sequences = load_sequences(
        sources,
        seq_path,
        meta_func,
        person_ids=person_ids,
    )
    logger.info(f"{len(sequences)} {dataset} sequences loaded")
    gait_features = list(gait_loader(feats_path).load_features())
    logger.info(f"{len(gait_features)} {dataset} features loaded")
    ds = Dataset(
        sequences,
        gait_features,
        dataset_name=dataset,
        seq_length=seq_length,
        normalization=norm,
        mirror=mirror,
    )
    logger.info(f"{len(ds)} samples in {dataset} dataset")
    return ds


def ds1_dataset_info(sources, cluster=False):
    if cluster:
        dataset_path = os.path.join(CLUSTER_BASE_PATH, "DS1")
    else:
        dataset_path = os.path.join(LOCAL_BASE_PATH, "DS1")
    metadata_func = get_ds1_metadata
    gait_loader = KinectLoader
    feats_path = os.path.join(dataset_path, "ds1_gait_features.csv")
    return dataset_path, sources, metadata_func, gait_loader, feats_path


def ds2_dataset_info(sources, cluster=False):
    if cluster:
        seq_path = os.path.join(CLUSTER_BASE_PATH, "ds2")
        feats_path = os.path.join(seq_path, "ds2_gait_fts.csv")
    else:
        ds_path = os.path.join(LOCAL_BASE_PATH, "DS2", "Analyses")
        seq_path = os.path.join(ds_path, "unzipped_final", "FINAL")
        feats_path = os.path.join(
            ds_path, "gait_fts", "ds2_gait_fts.csv"
        )
    sources = [s + "/raw" for s in sources]
    metadata_func = get_ds2_metadata
    gait_loader = DS2KinectLoader
    return seq_path, sources, metadata_func, gait_loader, feats_path
