# Cheetah Pose Estimation

This project was used in my MSc thesis topic, 'Monocular 3D Reconstruction of Cheetahs in the Wild'.

The code is not intended for use by external parties, as it currently relies on certain private packages that have not yet been made publicly available.

## Results
Video outputs of the pose estimation can be found in `data/video_results`. The `kinetic_dataset` folder contains multi-view 3D reconstructions of the cheetah, and compares kinematic (black skeleton) to physics-based reconstructions (orange skeleton).

The remaining folders contain monocular 3D reconstruction data, and are organised as
- `default`: kinematic reconstruction with an assumed constant acceleration motion model.
- `data-driven`: kinematic reconstruction with a statistically learned motion and pose model.
- `physics-based`: kinetic reconstruction with a physics-based motion model.
Note that the black skeleton denotes the ground truth and the orange skeleton the monocular estimate.

The `data/test_set` folder provides all reconstruction data in the same output format to [AcinoSet](https://github.com/African-Robotics-Unit/AcinoSet).
