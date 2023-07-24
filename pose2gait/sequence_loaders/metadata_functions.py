import csv
import logging
import re

logger = logging.getLogger(__name__)


def get_ds1_metadata(path):
    '''Get fps, walk_id, seq_id from first row'''
    metadata = {}
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        accessed = False
        for row in reader:
            accessed = True
            if 'fps' in row:
                fps = row['fps']
            else:
                # Note: openpose results do not have fps row
                # Thus the hard coded default
                fps = 30.0
            walk_name = row['walk_name'].lower()
            walk_name_segments = walk_name.split('_')
            person_index = walk_name_segments.index("id") + 1
            seq_id = walk_name_segments[-1]
            metadata["person_id"] = walk_name_segments[person_index]
            metadata["direction"] = "forward"
            if "state" in walk_name_segments:
                state_index = walk_name_segments.index("state") + 1
                metadata['state'] = walk_name_segments[state_index]
            walk_id = '_'.join(walk_name_segments[0:person_index + 1])
            break
    if not accessed:
        raise ValueError(f"File at {path} does not have any rows - cannot read metadata")
    return fps, walk_id, seq_id, metadata


def get_ds2_metadata(path):
    '''Get fps, walk_id, seq_id from first row'''
    metadata = {}
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        accessed = False
        for row in reader:
            accessed = True
            fps = row['fps']
            walk_id = row['walk_id'].lower()
            walk_name = row['walk_name'].lower()
            walk_name_segments = walk_name.split('_')
            seq_id = walk_name_segments[-1]
            metadata["person_id"] = row['patient_id']
            is_backward = not row['is_backward'] == 'false'
            metadata["direction"] = 'backward' if is_backward else 'forward'
            metadata['detector'] = row['detector']
            metadata['width'] = row['width']
            metadata['height'] = row['height']
            if "state" in walk_name_segments:
                state_index = walk_name_segments.index("state") + 1
                metadata['state'] = walk_name_segments[state_index]
            break
    if not accessed:
        raise ValueError(f"File at {path} does not have any rows - cannot read metadata")
    return fps, walk_id, seq_id, metadata
