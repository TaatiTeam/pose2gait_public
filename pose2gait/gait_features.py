import enum
import numpy as np
import logging
import csv

logger = logging.getLogger(__name__)


class GaitSource(enum.Enum):
    Zeno = 0
    Xsens = 1
    Kinect = 2
    Computed = 3
    Predicted = 4


class GaitFeatures:
    ''' A class to hold a set of gait features corresponding to
    a Sequence of Poses.

    Args:
        features:
        source (GaitSource):

    '''
    def __init__(self, walk_id, features, source, metadata=None):
        """ A class to hold a set of gait features for a walking bout
        TODO: Allow access using string keys like a dictionary
        Args:
            walk_id (String): A unique id for each walk. Should equal the
                walk_id for any PoseSequence objects that represent the same
                walking bout
            features (_type_): a dictionary from string feature names to float
                values
            source (GaitSource): an enum value representing the source that
                the features came from (e.g. XSens, Zeno, Kinect)
            metadata (dict, optional): Dictionary from string keys to values
                representing additional metadata such as direction of walk and
                person id. Defaults to None.
        """
        self.walk_id = walk_id
        self.source = source
        self.metadata = {} if metadata is None else metadata

        self.num_steps = None

        # temporal
        self.cadence = None  # steps per min
        self.avg_step_time = None  # in seconds
        self.cv_step_time = None
        self.si_step_time = None

        # spatiotemporal
        self.velocity = None  # cadence * step length = meters/minute?
        self.avg_step_width = None  # meters
        self.cv_step_width = None
        self.si_step_width = None
        self.avg_step_length = None  # meters
        self.cv_step_length = None
        self.si_step_length = None
        self.avg_MOS = None
        self.avg_min_MOS = None
        self.timeout_MOS = None

        # initialize
        self.init_features(features)

    def init_features(self, features):
        for name, val in features.items():
            if name in self.__dict__.keys() and val != "":
                self.__dict__[name] = float(val)
            else:
                logger.debug(f"Feature {name} not in GaitFeatures class"
                             " or no value provided")

    def to_numpy(self, names=False):
        arr = np.array([
            self.cadence,
            self.avg_step_time,
            self.cv_step_time,
            self.si_step_time,
            self.velocity,
            self.avg_step_width,
            self.cv_step_width,
            self.si_step_width,
            self.avg_step_length,
            self.cv_step_length,
            self.si_step_length,
            self.avg_MOS,
            self.avg_min_MOS,
            self.timeout_MOS,
            ], dtype=np.float32)

        if names:
            name_strings = [
                "cadence",
                "avg_step_time",
                "cv_step_time",
                "si_step_time",
                "velocity",
                "avg_step_width",
                "cv_step_width",
                "si_step_width",
                "avg_step_length",
                "cv_step_length",
                "si_step_length",
                "avg_MOS",
                "avg_min_MOS",
                "timeout_MOS",
            ]
            return arr, name_strings
        else:
            return arr

    def to_file(self, filename):
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.__dict__.keys())
            writer.writeheader()
            writer.writerow(self.__dict__)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                walk_id = row.pop("walk_id")
                source = row.pop("source")
                metadata = row.pop("metadata")
                features = row
                return GaitFeatures(walk_id, features, source,
                                    metadata=metadata)
