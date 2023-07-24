import random

import torch
from pose_sequence.utils import mirror_sequence
from torch.nn.functional import one_hot
from torch.utils.data import Dataset as torchDataset

SOURCE_DICT = {"openpose": 0, "alphapose": 1, "detectron": 2}

DATASET_DICT = {
    "ds1": 0,
    "ds2": 1,
}


def get_cv_folds_by_person(dataset, n=10):
    """Split people in dataset into n equal folds

    Args:
        dataset (pose2gait.Pose2GaitDataset): a pose2gait dataset containing
            walks from all individuals that you want to split for cross
            validation
        n (int, optional): The number of CV folds. Defaults to 10.

    Raises:
        ValueError: Raised if there are fewer than n people in the dataset

    Returns:
        list of list of string: A list of length n, where each element is a
            list of person ids in that fold.
    """
    participant_ids = list(dataset.person_ids)
    num_participants = len(participant_ids)
    if num_participants < n:
        raise ValueError(f"{num_participants} not enough people for {n} folds")
    fold_sizes = [num_participants // n] * n
    for i in range(num_participants % n):
        fold_sizes[i] = fold_sizes[i] + 1
    folds = []
    i = 0
    for size in fold_sizes:
        folds.append(participant_ids[i: i + size])
        i = i + size
    return folds


class Pose2GaitDataset(torchDataset):
    def __init__(
        self,
        dataset,
        by_walk=True,
        target_features=None,
        source_dict=SOURCE_DICT,
        dataset_dict=DATASET_DICT,
    ):
        """A torch dataset for training the pose2gait model. Takes in
        a pose2gait.Dataset object and formats it for pytorch access.

        Args:
            dataset (pose2gait.Dataset): _description_
            by_walk (bool, optional): If true, take only one sequence per walk
                per epoch. If false, use all sequences in each epoch.
                Defaults to True.
            target_features (list of str, defaults to None): Limit analysis to
                these target features. efaults to None, which uses all features
                in the GaitFeatures objects.
            source_dict (dict(string -> int), optional): A mapping from source
                algorithms to class labels to use for one-hot encoding of
                metadata. Defaults to {"openpose: 0", "alphapose": 1,
                "detectron": 2}.
            dataset_dict (dict(string -> int), optional): A mapping from
                dataset names to class labels to use for one-hot encoding of
                metadata. Defaults to {"mdc": 0, "tri": 1, "belmont": 2,
                "lakeside": 3}
        """
        self.dataset = dataset
        self.mirror = self.dataset.mirror
        self.by_walk = by_walk
        self.target_features = target_features
        self.source_dict = source_dict
        self.dataset_dict = dataset_dict

    def __len__(self):  
        if self.by_walk:
            # only take 1 sequence per walk per epoch
            size = len(self.dataset.walk_ids)
        else:
            # use all sequences per epoch
            size = len(self.dataset.sequences)
        
        if self.mirror:
            return size * 2
        else:
            return size

    def __getitem__(self, index):
        if self.by_walk:
            mirror = False # do we mirror this specific sequence
            if self.mirror and index > len(self.dataset.walk_ids):
                index = index - len(self.dataset.walk_ids)
                mirror = True
            walk_id = self.dataset.walk_ids[index]
            sequence = random.choice(self.dataset.pose_sequences_map[walk_id])       
        else:
            mirror = False
            if self.mirror and index > len(self.dataset.sequences):
                index = index - len(self.dataset.sequences)
                mirror = True
            sequence = self.dataset.sequences[index]
            walk_id = sequence.walk_id
        if mirror:
            sequence = mirror_sequence(sequence)
        features = self.dataset.gait_features_map[walk_id]
        sequence, features = self.dataset.preprocess(sequence, features)
        return self.seq_to_tensor(sequence), self.feat_to_tensor(features)

    def one_hot_source(self, sequence):
        """Get a one-hot encoding of the source algorithm for a given sequence.

        Args:
            sequence (pose_sequence.PoseSequence): A pose sequence with a
                source name included in it's seq_id.

        Raises:
            ValueError: If no source names from self.source_dict are included
            in the sequence.seq_id.

        Returns:
            torch.tensor: a one-hot encoding of the source algorithm
        """
        seq_id = sequence.seq_id.lower()
        index = None
        for key, val in self.source_dict.items():
            if key in seq_id:
                index = val
                break
        if index is None:
            source_keys = list(self.source_dict.keys())
            raise ValueError(
                f"None of source keys {source_keys} in seq_id {seq_id}"
            )
        return one_hot(torch.tensor(index), num_classes=len(self.source_dict))

    def one_hot_dataset(self, sequence):
        """Get a one-hot encoding of the dataset for a given sequence.

        Args:
            sequence (pose_sequence.PoseSequence): A pose sequence with the
                'dataset' key in it's metadata.

        Raises:
            ValueError: If the dataset name is not included in
                self.dataset_dict.

        Returns:
            torch.tensor: a one-hot encoding of the dataset
        """
        dataset = sequence.metadata["dataset"]
        if dataset not in self.dataset_dict:
            raise ValueError(
                f"Dataset {dataset} not in map {self.dataset_dict}"
            )
        index = self.dataset_dict[dataset]
        return one_hot(torch.tensor(index), num_classes=len(self.dataset_dict))

    def seq_to_tensor(self, seq):
        """Convert a pose sequence to a tensor for feeding into a model.
        Includes the joint locations and metadata indicating source algorithm
        and dataset.

        Args:
            seq (pose_sequence.PoseSequence): A pose sequence to turn into a
                tensor. Assumes 'dataset' is in metadata and the source name is
                included in the seq_id.

        Returns:
            (torch.tensor, torch.tensor): A tuple containing two tensors, the
                first containing joint locations and the second one-hot
                encodings of the dataset and source algorithm.
        """
        loc_tensor = torch.from_numpy(seq.to_numpy())
        metadata_tensor = torch.cat(
            (self.one_hot_dataset(seq), self.one_hot_source(seq))
        )
        return loc_tensor, metadata_tensor

    def feat_to_tensor(self, feat):
        """Convert a set of gait features to a tensor for feeding into a torch
        model. Optionally limits inclusion to the specified set of features in
        self.target_features.

        Args:
            feat (pose2gait.GaitFeatures): A set of gait features

        Returns:
            torch.tensor: A tensor containing the specified target gait
                features, or all features if a subset is not specified.
        """
        features_arr, feat_names = feat.to_numpy(names=True)
        if self.target_features:
            keep_indices = []
            for feat in self.target_features:
                keep_indices.append(feat_names.index(feat))
            features_arr = features_arr[keep_indices]
        return torch.from_numpy(features_arr)
