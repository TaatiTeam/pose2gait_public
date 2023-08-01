import argparse
import sys
import pandas as pd
import json
sys.path.append('..')
from pose2gait.experiments.load_datasets import get_ds1_metadata
from pose_sequence import PoseSequence
from gait_features_from_pose.compute_heel_strikes import ST_DBSCAN
from gait_features_from_pose.compute_heel_strikes.st_dbscan import DBSCAN_params
from gait_features_from_pose.compute_features import ComputeFeatures3D, compute_temporal_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="sample_kinect.csv", help="Directory to the input file.")
    parser.add_argument('-o', '--out', type=str, default="sample_out.json", help="path to the output file.")
    args = parser.parse_args()
    return args


def get_joint_columns(joint_names):
    joint_columns = []
    for joint_name in joint_names:
        joint_columns.append(f"{joint_name}_x")
        joint_columns.append(f"{joint_name}_y")
        joint_columns.append(f"{joint_name}_z")
    return joint_columns


def read_pose(path, joint_names):
    df = pd.read_csv(path)
    joint_columns = get_joint_columns(joint_names)
    num_joints = len(joint_names)
    joint_info = df[joint_columns].values.reshape(-1, num_joints, 3)
    return joint_info


def remove_close_strikes(heel_strike_frames, threshold=5):
    """If two heel strikes are detected within 5 frames, it removes the second one."""
    result = [heel_strike_frames[0]]
    for i in range(1, len(heel_strike_frames)):
        if heel_strike_frames[i] - result[-1] >= threshold:
            result.append(heel_strike_frames[i])
    return result


def main():
    args = parse_args()
    joint_names = [
            'LShoulder', 'RShoulder',
            'LElbow', 'RElbow',
            'LWrist', 'RWrist',
            'Sacr',
            'LHip', 'RHip',
            'LKnee', 'RKnee',
            'LAnkle', 'RAnkle']
    joint_connections = [
            ('LShoulder', 'RShoulder'),
            ('LShoulder', 'LElbow'),
            ('LElbow', 'LWrist'),
            ('RShoulder', 'RElbow'),
            ('RElbow', 'RWrist'),
            ('LHip', 'Sacr'),
            ('Sacr', 'RHip'),
            ('LHip', 'LKnee'),
            ('LKnee', 'LAnkle'),
            ('RHip', 'RKnee'),
            ('RKnee', 'RAnkle'),
        ]
    
    frame_rate, walk_id, seq_id, metadata = get_ds1_metadata(args.input)
    joint_info = read_pose(args.input, joint_names)

    sequence = PoseSequence(
            walk_id, seq_id, frame_rate,
            joint_names, joint_connections,
            dims=3,
            joint_info=joint_info,
            metadata=metadata, pose_func=lambda: read_pose(args.input, joint_names)
            )
    _, detected_heel_strikes = ST_DBSCAN.compute_heel_strikes(sequence,
                                                              dbscan_params=DBSCAN_params(min_pts=1, eps_spatial=0.30))
    start_is_left = detected_heel_strikes[0][1] == 'L'
    heel_strike_frames = [frame for frame, foot_label in detected_heel_strikes]
    heel_strike_frames = remove_close_strikes(heel_strike_frames)
    
    compute_features_3d = ComputeFeatures3D(sequence, heel_strike_frames, start_is_left, sacrum_name='Sacr')
    spatiotemproal_features = compute_features_3d.compute_spatiotemporal_features()
    temporal_features = compute_temporal_features(heel_strike_frames, fps=30)
    out = {
        'step_width': spatiotemproal_features['avg_step_width'],
        'step_length': spatiotemproal_features['avg_step_length'],
        'velocity': spatiotemproal_features['velocity'],
        'step_time': temporal_features['avg_step_time']
    }
    with open(args.out, 'w') as fp:
        json.dump(out, fp)
    

if __name__ == '__main__':
    main()