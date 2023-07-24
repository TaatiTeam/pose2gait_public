from pose_sequence.visualization import visualize_sequence_matplotlib
import os
import logging

logger = logging.getLogger(__name__)


def visualize_dataset(dataset, outdir, max_videos=None, min_conf=None):
    logger.info(f"Writing {max_videos} videos to f{outdir}")

    i = 0
    for seq, feat in dataset.get_data():
        name = f"{seq.walk_id}_{seq.seq_id}.avi"
        visualize_sequence_matplotlib(
            seq,
            os.path.join(outdir, name),
            exclude_joints=lambda x: x[0] < min_conf,
            gait_features=feat)
        i += 1
        if max_videos is not None and i > max_videos:
            break
