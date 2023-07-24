from .gait_loader import GaitLoader
from pose2gait import GaitFeatures, GaitSource
from csv import DictReader
import logging

logger = logging.getLogger(__name__)


class KinectLoader(GaitLoader):
    def __init__(self, filename):
        super().__init__(filename)
        self.name_map = {
            'Walking speed (m/s)': 'velocity',
            'Cadence': 'cadence',
            'Step time (s)': 'avg_step_time',
            'Step width (m)': 'avg_step_width',
            'Step length (m)': 'avg_step_length',
            'CV step time': 'cv_step_time',
            'CV step length' :'cv_step_length',
            'CV step width' :'cv_step_width',
            'Symmetry step time': 'si_step_time',
            'Symmetry step width': 'si_step_width',
            'Symmetry step length': 'si_step_length',
            'MOS average': 'avg_MOS',
            'MOS minimum': 'avg_min_MOS',
            }

    def _load_features(self, path):
        source = GaitSource.Zeno
        with open(path, 'r') as f:
            reader = DictReader(f)
            for row in reader:
                metadata = {}
                walk_id = row.pop('WalkID').lower()
                split = walk_id.split("_")
                person_index = split.index("id") + 1
                metadata['person_id'] = split[person_index]
                count = 0
                if "state" in split:
                    state_index = split.index("state") + 1
                    count_index = state_index + 1
                    metadata['state'] = split[state_index]
                    if count_index < len(split):
                        count = split[count_index]
                        try:
                            count = int(count)
                        except:
                            pass
                metadata['count'] = count

                if "cued" in split:
                    cued_index = split.index("cued") + 1
                    metadata['cued'] = split[cued_index]
                walk_id = '_'.join(split[0:person_index + 1])
                metadata['walk_id'] = walk_id
                metadata["full_walk_id"] = row.pop("Full WalkID")
                metadata["amb_id"] = row.pop("AMBID")
                metadata['direction'] = "forward"
                
                for old_name, new_name in self.name_map.items():
                    row[new_name] = row.pop(old_name)
                yield GaitFeatures(walk_id, row, source, metadata)