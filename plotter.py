import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import argparse
import cv2
import json
import h5py
from pathlib import Path
from ffprobe import FFProbe
from tqdm import tqdm
import dateutil
import datetime


class Plotter:

    def __init__(self, video_file, log_file, intr_calib_file, extr_calib_file, output_file, trail_len=20):
        self.trail_len = trail_len

        self.cam_mtx, self.dist_coeffs, self.rvec, self.tvec = self.import_cam_calib(extr_calib_file, 
                                                                                     intr_calib_file,
                                                                                     video_file)
        # Input video file
        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"{video_file} does not exist or is not a video file.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.t_start_video = self.get_video_startime(video_file)
        self.trajectories = self.import_trajectories(log_file)
        self.frame_id = 0
        self.t_offset = 0

        # Output video file
        if output_file is None:
            video_file = Path(video_file)
            video_file_name = video_file.stem
            video_file_dir = video_file.parent
            self.output_file = video_file_dir / f"{video_file_name}_overlay.mp4"
        else:
            self.output_file = output_file

        # Plotting settings
        dpi = mpl.rcParams['figure.dpi']
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        figsize = self.width / float(dpi), self.height / float(dpi)
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.cmap = plt.get_cmap("tab20")  # Colormap for 20 drones

        # Build trajectories for plotting
        self.t_frames = np.arange(self.n_frames) / self.fps + self.t_start_video  # Initial guess of absolute timestamp of frames

        # Window and callbacks
        self.window_name = "Video"
        self.frame_slider = "Frame"
        self.offset_slider = "Offset"
        self.offset_range_secs = 10  # Adjustable offset +- 10 seconds
        self.offset_range = int(self.fps) * 2 * self.offset_range_secs  # Offset in integers for cv2 slider

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(self.frame_slider, self.window_name, 0, self.n_frames - 1, self.set_frame)
        cv2.createTrackbar(self.offset_slider, self.window_name, int(self.offset_range / 2), self.offset_range, self.set_offset)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def import_trajectories(self, log_file):
        """Loads and returns a list of trajectories from a log file."""
        trajectories = []
        with h5py.File(log_file, 'r') as f:
            for i, group in f.items():
                trajectories.append(np.array(group['pos']))

        return trajectories 

    def import_cam_calib(self, extr_calib_file, intr_calib_file, video_file):
        """Import extrinsic and intrinsic calibrations from json files."""
        dir_path = Path(video_file).parent
        if extr_calib_file is None:
            extr_calib_file = dir_path / "extrinsic_calib.json" 
        if intr_calib_file is None:
            intr_calib_file = dir_path / "intrinsic_calib.json" 

        with open(extr_calib_file) as f:
            extrinsic_calib = json.load(f)
            rvec = np.array(extrinsic_calib['rvec'])
            tvec = np.array(extrinsic_calib['tvec'])

        with open(intr_calib_file) as f:
            intrinsic_calib = json.load(f)
            cam_mtx = np.array(intrinsic_calib['cam_mtx'])
            dist_coeffs = np.array(intrinsic_calib['dist_coeffs'])

        return cam_mtx, dist_coeffs, rvec, tvec

    def get_video_startime(self, video_file):
        """Estimate start date of video file using its metadata.
        Note that this method might give incorrect results depending on the recording device. Only tested
        for videos recorded on Android."""
        metadata = FFProbe(video_file)
        start_time_str = metadata.metadata['creation_time']
        duration_str = metadata.metadata['Duration']
        if 'Z' not in start_time_str:
            print("Warning: Video timestamp not in UTC")
        start_time = dateutil.parser.isoparse(start_time_str)  # Start time of video in UTC
        duration_t = dateutil.parser.parse(duration_str)  # Start time of video in UTC
        duration = datetime.timedelta(hours=duration_t.hour, minutes=duration_t.minute, seconds=duration_t.second)
        start_time_in_secs = start_time.timestamp() - duration.total_seconds()

        return start_time_in_secs

    def set_frame(self, frame_id):
        """Callback method for frame slider."""
        self.frame_id = frame_id
        self.update()

    def set_offset(self, offset):
        """Callback method for offset slider."""
        self.t_offset = (offset / int(self.fps)) - self.offset_range_secs  # Convert slider value to offset in secs
        self.update()

    def update(self):
        """Update displayed image with overlaid trajectories."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)  # Get current frame
        ret, frame = self.cap.read()

        # Plot trajectories over frame
        trajectories_proj = self.get_trajectories_proj()
        out_frame = self.plot(frame, trajectories_proj, self.frame_id, self.trail_len)
        cv2.imshow(self.window_name, out_frame)

    def get_trajectories_proj(self):
        """Compute projection of 3D trajectories onto image plane."""
        t_frames_shifted = self.t_frames + self.t_offset
        trajectories_proj = []
        # Plot trajectory of each drone
        for trajectory in self.trajectories:
            t, pos = trajectory[:, 0], trajectory[:, 1:]  #  3D trajectory: [t, x, y, z]
            pos_interp = np.array([np.interp(t_frames_shifted, t, pos[:, i]) for i in range(pos.shape[1])]).T
            img_points, _ = cv2.projectPoints(pos_interp, self.rvec, self.tvec, self.cam_mtx, self.dist_coeffs)
            img_points = img_points.squeeze()
            trajectories_proj.append(img_points)

        return trajectories_proj

    def plot(self, img, trajectories, frame_id, trail_len):
        """Plot projections of trajectories over image."""
        # TODO: Make this more object oriented (no arguments) or completely functional (pass fig and ax as arguments).

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Change color space for matplotlib plotting

        # Prepare for plotting
        self.ax.clear()
        self.ax.axis('off')
        self.ax.imshow(img, interpolation='nearest')  # This takes too long...

        # Handle case where not enough points for full trail length
        if frame_id > trail_len:
            plot_range = np.arange(frame_id - trail_len, frame_id)
        else:
            plot_range = np.arange(0, frame_id)

        line_widths = np.linspace(0.5, 3, trail_len)  # Same line width for all drones

        for i, traj in enumerate(trajectories):
            color = self.cmap(i / 20)
            line_colors = np.linspace((*color[:3], 0), color, trail_len, axis=0)   # Fade trail color to transparent
            traj = traj[plot_range, np.newaxis, :]
            segments = np.concatenate([traj[:-1], traj[1:]], axis=1)
            lines = LineCollection(segments, linewidths=line_widths, colors=line_colors)
            lines.set_capstyle('round')
            self.ax.add_collection(lines)

        # Retrieve view of render buffer. See: https://stackoverflow.com/a/62040123
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        out_frame = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGB2BGR)

        return out_frame

    def run(self):
        """Main program loop."""
        self.update()  # Initialize screen with first frame
        while cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(500)
            if keyCode == ord('k'):
                break
        cv2.destroyAllWindows()
    
    def save(self):
        """Save video with overlaid trajectories as adjusted by the user."""
        out = cv2.VideoWriter(str(self.output_file), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        pbar = tqdm(total=self.n_frames)  # Progress bar

        frame_id = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # Reset to start
        trajectories_proj = self.get_trajectories_proj()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            out_frame = self.plot(frame, trajectories_proj, frame_id, self.trail_len)
            out.write(out_frame)
            frame_id += 1
            pbar.update(1)

        pbar.close()
        self.cap.release()
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trajectories over a video')
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('log_file', type=str, help='Path to log file with trajectories')
    parser.add_argument('--intrinsic_calib', type=str, help='Path to intrinsic calibration file')
    parser.add_argument('--extrinsic_calib', type=str, help='Path to extrinsic calibration file')
    parser.add_argument('--output', type=str, help='Output video file (mp4) with overlaid trajectories')
    args = parser.parse_args()

    video_file = args.video_file
    log_file = args.log_file
    intr_calib_file = args.intrinsic_calib
    extr_calib_file = args.extrinsic_calib
    output_file = args.output

    plotter = Plotter(video_file, log_file, intr_calib_file, extr_calib_file, output_file, trail_len=30)
    plotter.run()
    plotter.save()