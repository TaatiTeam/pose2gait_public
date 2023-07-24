import logging

import numpy as np
from gait_features_from_pose.utils import interpolate_and_filter

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, sequences, features,
                 seq_length=120, exclude_multiples=True, dataset_name=None,
                 normalization="per_video", mirror=False):
        """A dataset containing pose sequences and corresponding gait features

        # TODO: make iterable without calling get_data

        Args:
            sequences (list of pose2gait.PoseSequence) : NOTE: cannot be generators!!
            features (list of pose2gait.GaitFeatures)
            seq_length (int) : Length of sequence to return in frames. Defaults to 120
            normalization (str): Type of normalization to perform. Valid values are:
                "per_video", "per_frame", "none". Defaults to "per_video." 
            mirror (bool, optional): If true, add a copy of each sequence to the
                dataset that is mirrored (flipped left and right joints
                and multiply x by -1).
        """
        self.pose_sequences_map = {}  # map from walk_id to lists of pose sequences
        self.gait_features_map = {}  # map from walk_id to gait features
        self.seq_length = seq_length
        self.person_ids = set()
        self.exclude_joints = ["LEar", "REar"]
        self.exclude_multiples = exclude_multiples
        self.dataset_name = dataset_name
        supported_norm = ["none", "per_video", "per_frame"]
        if normalization not in supported_norm:
            raise ValueError(
                f"Provided normalization {normalization} not in supported methods {supported_norm}")
        self.normalization = normalization
        self.mirror = mirror

        # load all sequences and features into maps
        # TODO: all the pre-processing is quite slow, need to speed it up
        for sequence in sequences:
            self.temporal_crop(sequence)
            sequence = sequence.filter_joints(self.exclude_joints)
            if self.check_valid_sequence(sequence):
                if self.dataset_name is not None:
                    sequence.metadata['dataset'] = self.dataset_name
                if sequence.walk_id not in self.pose_sequences_map:
                    self.pose_sequences_map[sequence.walk_id] = []
                self.pose_sequences_map[sequence.walk_id].append(sequence)

        for feature_set in features:
            if self.exclude_multiples:
                if "count" in feature_set.metadata and feature_set.metadata["count"] > 1:
                    continue
            if feature_set.walk_id in self.gait_features_map:
                logger.error(f"Attempted to add duplicate walk_id {feature_set.walk_id} to gait feature set - skipping")
                continue
            self.gait_features_map[feature_set.walk_id] = feature_set

        # filter sequences and gait features to shared walk ids
        seq_walk_ids = set(self.pose_sequences_map.keys())
        gait_walk_ids = set(self.gait_features_map.keys())
        self.walk_ids = list(seq_walk_ids & gait_walk_ids)
        seq_only_ids = list(seq_walk_ids - gait_walk_ids)
        gait_only_ids = list(gait_walk_ids - seq_walk_ids)
        logger.debug(f"{len(seq_only_ids)} sequence walk ids not in features: "\
            f"{seq_only_ids if len(seq_only_ids) < 10 else seq_only_ids[0:10]}")
        logger.debug(f"{len(gait_only_ids)} feature walk ids not in sequences: "\
            f"{gait_only_ids if len(gait_only_ids) < 10 else gait_only_ids[0:10]}")

        self.sequences = []
        for walk_id in list(self.pose_sequences_map.keys()):
            if walk_id not in self.walk_ids:
                self.pose_sequences_map.pop(walk_id)
            else:
                self.sequences.extend(self.pose_sequences_map[walk_id])
                self.person_ids.add(self.pose_sequences_map[walk_id][0].metadata["person_id"])

        self.features = []
        for walk_id in list(self.gait_features_map.keys()):
            if walk_id not in self.walk_ids:
                self.gait_features_map.pop(walk_id)
            else:
                self.features.append(self.gait_features_map[walk_id])

    def __len__(self):
        return len(self.walk_ids)

    def temporal_crop(self, sequence):
        # crop to length
        if sequence.metadata['direction'] == "forward":
            start_ind = max(0, sequence.num_frames-self.seq_length)  # start seq_length from end
            end_ind = sequence.num_frames  # go to end
        elif sequence.metadata['direction'] == "backward":
            start_ind = 0  # start at beginning of video
            end_ind = min(self.seq_length, sequence.num_frames)  # go to seq_length
        else:
            raise ValueError("Expected dirction forward or backward,"\
                            f" got {sequence.metatdata['direction']}")
        cropped_info = sequence.joint_info[start_ind:end_ind]
        sequence.set_joint_info(cropped_info)

        # pad to length by repeating last pose
        if sequence.num_frames < self.seq_length:
            joint_info = sequence.joint_info
            repeat_info = [joint_info[-1]] * (self.seq_length - sequence.num_frames)
            sequence.set_joint_info(np.concatenate((joint_info, repeat_info), axis=0))

        assert sequence.num_frames == self.seq_length

    def check_valid_sequence(self, sequence):
        """Reasons for invalid sequence:
            - More than half of the values are missing/nan
            - The sequence is too short (less than self.seq_length)"""
        joint_locs = sequence.to_numpy()
        num_missing = np.count_nonzero(np.isnan(joint_locs))
        if num_missing > joint_locs.size / 2:
            logger.warn(f"Removing sequence {sequence.walk_id}:{sequence.seq_id}: "
                        + f"{num_missing}/{joint_locs.size} values missing.")
            return False
        if len(joint_locs) < self.seq_length:
            logger.warn(f"Removing sequence {sequence.walk_id}:{sequence.seq_id}: "
                        + f"only has {len(joint_locs)} frames out of {self.seq_length} required.")
            return False
        return True

    def preprocess(self, sequence, features):
        interpolate_and_filter(sequence, interpolate_only=True)
        # normalize by hip distance/location in middle frame
        self.normalize(sequence)
        return sequence, features

    def normalize(self, sequence, hip_names=('LHip', 'RHip')):
        """Normalize provided sequence in-place using stored normalizaiton strategy.
        If "none", do nothing. If "per_video", shift and scale the whole sequence
        so that in the center frame, the point between the hips is at 0, 0 and the inter-hip
        distance is 1. If "per_frame", for each frame, shift and scale to center mid-hip 
        point at 0, 0 and set inter-hip distance to 1.
        """
        if self.normalization == "none":
            pass
        elif self.normalization == "per_video":
            target_frame = sequence.num_frames //2
            shift, scale = self.__get_shift_and_scale(sequence, target_frame, hip_names=hip_names)
            sequence.shift(shift)
            sequence.scale(scale)
        elif self.normalization == "per_frame":
            joint_locs = sequence.get_joint_locations()
            for frame in range(sequence.num_frames):
                shift, scale = self.__get_shift_and_scale(sequence, frame, hip_names=hip_names)
                joint_locs[frame] = (joint_locs[frame] + shift) * scale
            sequence.set_joint_locations(joint_locs)

    def __get_shift_and_scale(self, sequence, frame, hip_names=('LHip', 'RHip')):
        """Compute the amount to shift and scale a pose to make the center of the hips
        located at (0, 0) and the inter-hip distance equal to 1.

        Args:
            sequence (PoseSequence): pose sequence to get pose from
            frame (int): Time frame to get pose from
            hip_names (tuple, optional): Names of hip joints. Defaults to ('LHip', 'RHip').
        
        Returns:
            (numpy.array, float): A tuple contaning the amount to shift the pose in x, y
                and the amount to scale the pose in all dims. If the hips are at the 
                same point, return 1 for the scale factor (no scaling).
        """
        hip1 = sequence.location_by_name(hip_names[0])[frame]
        hip2 = sequence.location_by_name(hip_names[1])[frame]
        shift = -1 *(hip1 + hip2) / 2
        if np.linalg.norm(hip1 - hip2) > 0:
            scale = 1/np.linalg.norm(hip1 - hip2)
        else:
            # if hips are equal, don't scale
            scale = 1
        return shift, scale      

    def get_data(self):
        """ Get the paired pose sequences and gait features in
        this dataset. The order is arbitrary.

        Yields:
            Tuple(PoseSequence, GaitFeatures):
                Pairs of PoseSequence and GaitFeature objects that
                have the same walk_id
        """
        for walk_id, pose_sequences in self.pose_sequences_map.items():
            gait_features = self.gait_features_map[walk_id]
            for sequence in pose_sequences:
                yield self.preprocess(sequence, gait_features)

    def get_subset_by_person(self, person_ids):
        new_sequences = [seq for seq in self.sequences if seq.metadata["person_id"] in person_ids]
        logger.debug(f"Subset contains {len(new_sequences)} sequences")
        features = self.features
        return Dataset(new_sequences, features, seq_length=self.seq_length, dataset_name=self.dataset_name)