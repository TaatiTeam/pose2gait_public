from abc import ABC, abstractmethod
import os


class GaitLoader(ABC):
    def __init__(self, filename):
        """An abstract class for loading gait features from a file.

        Args:
            base_path (str): a path to the file containing the gait features
        """
        self.filename = filename

        assert os.path.isfile(self.filename),\
            f"Given filename {self.filename} is not a file"
    
    def __len__(self):
        """Number of GaitFeatures objects that the loader can produce.
        By default, returns the number of lines in the file,
        minus 1 for the header, but can be overwritten in subclasses.

        Returns:
            int: length of the loader
        """
        with open(self.filename, 'r') as f:
            num_lines = sum(1 for _ in f)
        return num_lines - 1
    
    def load_features(self):
        """Return the gait features in the stored file lazily.

        Yields:
            GaitFeatures: gait features loaded from the file
        """
        for feat in self._load_features(self.filename):
            yield feat

    @abstractmethod
    def _load_features(self, path):
        """Load the gait features in a specific file. Must be implemented
        in subclasses.

        Args:
            path (str): path to file containing gait features
        
        Yields:
            GaitFeatures: gait features loaded from the file at path

        Raises:
            NotImplementedError: If not implemented in subclass
        """
        raise NotImplementedError("_load_features called but not implemented")
