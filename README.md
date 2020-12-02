# trajectory-overlay

A collection of scripts to overlay videos with trajectories from a motion capture system.

**Note**: This is still heavily work in progress.

## Instructions

### 1. Intrinsic calibration
If you use a camera for which no intrinsic calibration file is provided, you will need
to record a video of the calibration pattern in `data`. The script `intrinsic_calibration.py`
will guide you through the calibration process and save the results into a `json` file. 

Note that the intrinsic calibration parameters depend on the image resolution. Make sure you
use the same settings as you will use to record the video on which trajectories will be drawn.

### 2. Extrinsic calibration
To determine the pose of the camera in the world frame, you will need to place at least 6 markers in the motion capture arena and write their positions into a `json` file as follows:
```json
{ 
    "pos": [[-1.459,  0.924, 0.014],
            [ 0.891,  1.781, 0.007],
            [ 3.049,  1.801, 0.011],
            [ 3.513, -1.529, 0.014],
            [ 1.262, -1.794, 0.007],
            [-1.124, -1.865, 0.010]]
}
```
Take a video of this setup and **do not** move the camera/phone anymore. Use the script
`extrinsic_calibration.py` to compute and save the extrinsic calibration into a `json` file. Make sure to visually check the reprojection of the markers (green). If they are totally off 
the intrinsic calibration is likely to be inaccurate. 

You can call `extrinsic_calibration.py` as follows:

```python3 extrinsic_calibration.py <video_path> <markers_file_path> <intrinsic_calib_file_path> ```

and add the following argument:

```--output <output_folder>```

### 3. Overlay plot
Finally you can use `plotter.py` to overlay trajectories on a video. The script expects the
trajectories to be saved in a `hdf5` file. This `hdf5` file should contain a separate group
for each trajectory (each drone/vehicle). Each group should contain a dataset called `pos`.
This dataset should be a `N x 4` array where each row consists of `[unix timestamp, x, y, z]`.

You can call `plotter.py` as follows:

```python3 plotter.py <video_path> <log_trajectories_path> <intrinsic_calib_file_path> ```

and add the following arguments:

```--intrinsic_calib <intrinsic_calib_file_path> --extrinsic_calib <extrinsic_calib_file_path> --output <output_folder>```
