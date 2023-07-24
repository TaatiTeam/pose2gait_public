from setuptools import setup

setup(
        name='pose2gait',
        version='0.0',
        packages=[
            'pose2gait',
            'pose2gait.sequence_loaders',
            'pose2gait.gait_loaders',
            'pose2gait.visualization',
            'pose2gait.evaluation',
            'pose2gait.torch'
        ],
)
