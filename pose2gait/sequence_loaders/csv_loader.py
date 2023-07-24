from .sequence_loader import SequenceLoader
from pose_sequence import PoseSequence
import csv
import os
import logging
import numpy as np
logger = logging.getLogger(__name__)


class CSVLoader(SequenceLoader):
    # Note: sometimes missing values are 0.0. Sometimes NaN.
    def __init__(self, base_path, metadata_func, dims=["x", "y"]):
        """_summary_

        Args:
            base_path (_type_): _description_
            metadata_func (_type_): A 1 argument function that takes in a filepath
                and returns fps, walk_id, seq_id, and any extra metadata for the walk.
            dims (list, optional): _description_. Defaults to ["x", "y"].
        """
        super().__init__(base_path)
        # ignore eyes and nose - not included in most results anyway
        self.joint_names = [
            'LEar', 'REar',
            'LShoulder', 'RShoulder',
            'LElbow', 'RElbow',
            'LWrist', 'RWrist',
            'LHip', 'RHip',
            'LKnee', 'RKnee',
            'LAnkle', 'RAnkle']
        self.joint_connections = [
            ('LShoulder', 'RShoulder'),
            ('LShoulder', 'LElbow'),
            ('LElbow', 'LWrist'),
            ('RShoulder', 'RElbow'),
            ('RElbow', 'RWrist'),
            ('LHip', 'RHip'),
            ('LHip', 'LKnee'),
            ('LKnee', 'LAnkle'),
            ('RHip', 'RKnee'),
            ('RKnee', 'RAnkle'),
        ]
        self.dims = dims
        self.metadata_func = metadata_func

    def _read_poses(self, path):
        """Read the poses in the csv file at the given path.

        Args:
            path (string): Path to csv file containing pose sequence

        Returns:
            (np.array): An FxJx(D+1) array
                representing joint locations plus a confidence value,
                where F is the number of frames, J is number of joints,
                and D is number of dimensions
        """
        all_info = []  # list of frames
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frame_info = []  # list of joints within a frame
                for joint in self.joint_names:
                    # get location
                    joint_info = []  # one joint in one frame (dimensions plus conf)
                    for dim in self.dims:
                        fieldname = f"{joint}_{dim}"
                        try:
                            joint_info.append(float(row[fieldname]))
                        except ValueError:
                            joint_info.append(np.nan)
                    # get confidence measure
                    fieldname = f"{joint}_conf"
                    joint_info.append(row[fieldname])
                    frame_info.append(joint_info)
                all_info.append(frame_info)
        return self.__numpy_float_converter(all_info, dims=3, nan_vals=[""])

    def __numpy_float_converter(self, arr, dims=3, nan_vals=['']):
        ''' Takes in a python list (of lists), replaces specified values with NaNs,
        and returns a numpy array of dtype float32.
        Args:
            arr (list): a python list with the specified number of dimensions
            dims (int): number of dimensions in the list.
                Valid values are 1, 2, and 3, with default to 3
            nan_vals (list): values that should be replaced with NaN.
                Default is empty string only'''
        if type(arr) is np.array:
            return arr
        if dims == 1:
            arr = ['NaN' if e in nan_vals else e for e in arr]
        elif dims == 2:
            for i, entry in enumerate(arr):
                entry = ['NaN' if e in nan_vals else e for e in entry]
                arr[i] = entry
        elif dims == 3:
            for i, row in enumerate(arr):
                for j, col in enumerate(row):
                    col = ['NaN' if e in nan_vals else e for e in col]
                    arr[i][j] = col
        else:
            logger.warn("Only 1D and 2D arrays supported")
        return np.array(arr, dtype=np.float32)

    def _load_sequence(self, path, lazy=False):
        logger.debug(f"Loading sequence at path {path}")
        try:
            frame_rate, walk_id, seq_id, metadata = self.metadata_func(path)
        except (ValueError):
            # no metadata found for this path - return None
            return None
        if lazy:
            joint_info = None
        else:
            joint_info = self._read_poses(path)

        return PoseSequence(
            walk_id, seq_id, frame_rate,
            self.joint_names, self.joint_connections,
            dims=len(self.dims),
            joint_info=joint_info,
            metadata=metadata, pose_func=lambda: self._read_poses(path))

    def _get_paths(self):
        paths = []
        for entry in os.listdir(self.base_path):
            path = os.path.join(self.base_path, entry)
            if os.path.isfile(path) and path.endswith('csv'):
                paths.append(path)

        return paths
