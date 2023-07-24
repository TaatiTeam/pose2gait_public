from .gait_loader import GaitLoader
from pose2gait import GaitFeatures, GaitSource
from csv import DictReader


class DS2KinectLoader(GaitLoader):
    def __init__(self, filename):
        super().__init__(filename)
        self.name_map = {
            'speed': 'velocity',
            'steptime_mean': 'avg_step_time',
            'CVSteptime': 'cv_step_time',
            'symSteptime': 'si_step_time',
            'stepwidth_mean': 'avg_step_width',
            'CVStepwidth': 'cv_step_width',
            'symStepWidth': 'si_step_width',
            'steplength_mean': 'avg_step_length',
            'CVSteplength': 'cv_step_length',
            'symStepLength': 'si_step_length',
            'MOS_mean_final_new': 'avg_MOS',
            'MOS_min_final_new': 'avg_min_MOS',
            }

    def _load_features(self, path):
        source = GaitSource.Kinect
        with open(path, 'r') as f:
            reader = DictReader(f)
            for row in reader:
                metadata = {}
                walk_id = row.pop('walk').lower().strip()
                person_index = row.pop('patient').strip()
                metadata['person_id'] = person_index
                split = walk_id.split("_")
                if "state" in split:
                    state_index = split.index("state") + 1
                    metadata['state'] = split[state_index]

                metadata['walk_id'] = walk_id
                metadata['direction'] = row.pop('direction')
                metadata['stepsofwalk'] = row.pop('stepsofwalk')
                metadata['start_frame'] = row.pop('start_frame')
                metadata['num_frames'] = row.pop('num_frames_in_trajectory')

                for old_name, new_name in self.name_map.items():
                    row[new_name] = row.pop(old_name)
                yield GaitFeatures(walk_id, row, source, metadata)
