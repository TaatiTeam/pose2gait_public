import torch
import torch.nn as nn
from pose2gait.torch import Pose2GaitModel, ModelConfig, Pose2GaitDataset
from pose_sequence import PoseSequence
from pose2gait.sequence_loaders import CSVLoader
from pose2gait.sequence_loaders.metadata_functions import get_ds1_metadata
import numpy as np
import json
import pandas as pd

def extract_alphapose_data():
    '''
    AlphaPose keypoints are stored in a json file, which must be opened
    and stored in a format that can be used by pose2gait (csv)
    '''
    alphapose_path = r"C:\Users\serge\Downloads\alphapose-results.json"
    with open(alphapose_path) as f:
        sequence = json.load(f)
    f.close()

    csv_dict = []
    header = ['time', 'Nose_x',	'Nose_y', 'Nose_conf', 'LEye_x', 'LEye_y', 'LEye_conf',	
              'REye_x', 'REye_y', 'REye_conf', 'LEar_x', 'LEar_y', 'LEar_conf',	
              'REar_x',	'REar_y', 'REar_conf', 'LShoulder_x', 'LShoulder_y', 'LShoulder_conf',	
              'RShoulder_x', 'RShoulder_y',	'RShoulder_conf', 'LElbow_x', 'LElbow_y', 'LElbow_conf',	
              'RElbow_x', 'RElbow_y', 'RElbow_conf', 'LWrist_x', 'LWrist_y', 'LWrist_conf',	
              'RWrist_x', 'RWrist_y', 'RWrist_conf', 'LHip_x', 'LHip_y', 'LHip_conf', 'RHip_x',	'RHip_y', 'RHip_conf',	
              'LKnee_x', 'LKnee_y',	'LKnee_conf', 'RKnee_x', 'RKnee_y', 'RKnee_conf', 'LAnkle_x', 'LAnkle_y', 'LAnkle_conf',	
              'RAnkle_x', 'RAnkle_y', 'RAnkle_conf', 'x_min', 'y_min', 'x_max',	'y_max', 
              'walk_name', 'fps', 'start_frame']
    for i in range(len(sequence)):
        csv_dict.append({})
        csv_dict[i]['time'] = i / 30
        for j in range(51):
            csv_dict[i][header[j+1]] = sequence[i]['keypoints'][j]
        csv_dict[i]['x_min'] = min(sequence[i]['keypoints'][0::3])
        csv_dict[i]['x_max'] = max(sequence[i]['keypoints'][0::3])
        csv_dict[i]['y_min'] = min(sequence[i]['keypoints'][1::3])
        csv_dict[i]['y_max'] = max(sequence[i]['keypoints'][1::3])
        csv_dict[i]['walk_name'] = '2023_08_02__16_33_30_ID_00_state_2__alphapose'
        csv_dict[i]['fps'] = 30
        csv_dict[i]['start_frame'] = 1

    df = pd.DataFrame(csv_dict)
    df.to_csv('demos/sample_keypoints.csv', index=False)

def normalize(sequence, hip_names=('LHip', 'RHip')):
    target_frame = sequence.num_frames // 2
    shift, scale = get_shift_and_scale(sequence, target_frame, hip_names=hip_names)
    sequence.shift(shift)
    sequence.scale(scale)
    
def get_shift_and_scale(sequence, frame, hip_names):
    hip1 = sequence.location_by_name(hip_names[0])[frame]
    hip2 = sequence.location_by_name(hip_names[1])[frame]
    shift = -1 *(hip1 + hip2) / 2
    if np.linalg.norm(hip1 - hip2) > 0:
        scale = 1/np.linalg.norm(hip1 - hip2)
    else:
        # if hips are equal, don't scale
        scale = 1
    return shift, scale      

def infer_sequence():
    checkpoint_path = r"C:\Users\serge\Documents\Pose2Gait Inference\Checkpoint\best_model.pt" # saved state of trained pose2gait model
    sequence_dir = r"C:\Users\serge\\Documents\Pose2Gait Inference\Sequences" # pose sequence acquired from alphapose
    seq_loader = CSVLoader(base_path=sequence_dir, metadata_func=get_ds1_metadata)

    model_config = ModelConfig(num_features=4,
                            metadata_len=5,
                            input_frames=120,
                            input_joints=12,
                            input_dims = 2,
                            encoder_channels = [14, 8],
                            kernel_size = 3,
                            linear_features = [500, 250, 100])
    model = Pose2GaitModel(model_config)
    state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    sequences = []
    for sequence in seq_loader.load_sequences():
        if len(sequence.joint_info) < 120:
            print("Sequence is too short, expected length is 120 frames!")
            exit
        else:
            sequence = sequence.filter_joints(["LEar", "REar"])
            normalize(sequence)
            joints = torch.from_numpy(sequence.joint_info[-120:, :, :2])
        joints = joints[None]
        metadata = torch.zeros(1, 5) # dataset, source
        sequences.append((joints, metadata))
    model.load_state_dict(state['state_dict'])
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            predictions = model(sequence)
            with open("demos/sample_output.txt", "w") as f:
                f.write(f'Step time: {predictions[0][0]} s\nStep width: {predictions[0][1]} m\nStep length: {predictions[0][2]} m\nVelocity: {predictions[0][3]} m/s')
                print()
    
if __name__ == "__main__":
    extract_alphapose_data()
    infer_sequence()