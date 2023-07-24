from abc import ABC, abstractmethod
import os
import logging

logger = logging.getLogger(__name__)


class SequenceLoader(ABC):
    ''' An abstract class for loading sequences of poses from a base location
    in the file system.

    Args:
        base_path (str): a path to the base directory containing the sequences
    '''
    def __init__(self, base_path):
        self.base_path = base_path

        assert os.path.isdir(self.base_path),\
            f"Base path {self.base_path} not a directory"

    def __len__(self):
        return len(self._get_paths())

    def load_sequences(self, lazy=False):
        """Return the pose sequences in the base path.
        Args:
            lazy (bool): flag indicating whether or to load poses from file lazily.
            Defaults to false, which loads them at initialization.

        Yields:
            PoseSequence: pose sequence loaded from base path.
        """
        paths = self._get_paths()
        for path in paths:
            logger.debug(path)
            sequence = self._load_sequence(path, lazy=lazy)
            if sequence is None:
                continue
            else:
                yield sequence

    @abstractmethod
    def _load_sequence(self, path, lazy=False):
        ''' Load the pose sequence at a specific file path. Must be implemented
        in subclasses. Should return None if it is unable to load a sequence from path.
        Args:
            path (str): path to file containing pose sequence
            lazy (bool): flag indicating whether or to load poses from file lazily.
            Defaults to false, which loads them at initialization.
        Returns: a PoseSequence
        '''
        raise NotImplementedError("_load_sequence called but not implemented")

    def _get_paths(self):
        ''' Gets the paths to specific pose sequences inside the base directory.
        Default implementation lists all the files in the directory, but can be
        overwritten in subclasses.

        Returns: a list of strings that are valid paths
        '''
        paths = []
        for entry in os.listdir(self.base_path):
            path = os.path.join(self.base_path, entry)
            if os.path.isfile(path):
                paths.append(path)

        return paths
