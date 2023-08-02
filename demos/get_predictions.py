import torch
import torch.nn as nn
from pose2gait.torch import Pose2GaitModel, ModelConfig, Pose2GaitDataset
from pose_sequence import PoseSequence
from pose2gait.sequence_loaders import CSVLoader
from pose2gait.sequence_loaders.metadata_functions import get_ds1_metadata
import numpy as np

def normalize(sequence, hip_names=('LHip', 'RHip')):
    target_frame = sequence.num_frames //2
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
            print(predictions)
            print(f'Step time: {predictions[0][0]} s\nStep width: {predictions[0][1]} m\nStep length: {predictions[0][2]} m\nVelocity: {predictions[0][3]} m/s')
            print()
    
if __name__ == "__main__":
    infer_sequence()