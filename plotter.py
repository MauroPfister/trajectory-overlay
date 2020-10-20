from matplotlib import pyplot as plt
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
from matplotlib import pyplot as plt
import matplotlib as mpl
import datetime


class Plotter:

    def __init__(self, video_file, log_file, trail_len=20):
        self.video_file = video_file
        self.log_file = log_file
        self.trail_len = trail_len

        self.dir_path = Path(video_file).parent

        self.cam_mtx, self.dist_coeffs, self.rvec, self.tvec = self.import_cam_calib(self.dir_path)
        self.t_start_video = self.get_video_startime(self.video_file)
        self.cf_pos = self.import_trajectories(self.log_file)
        self.frame_id = 0
        self.t_offset = 0

        # Input video capture
        self.cap = cv2.VideoCapture(self.video_file)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        self.offset_range = int(self.fps) * 2 * 10

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(self.frame_slider, self.window_name, 0, self.n_frames - 1, self.set_frame)
        cv2.createTrackbar(self.offset_slider, self.window_name, int(self.offset_range / 2), self.offset_range, self.set_offset)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def import_trajectories(self, log_file):
        cf_pos = []
        with h5py.File(log_file, 'r') as f:
            for cf_id, cf_group in f.items():
                cf_pos.append(np.array(cf_group['pos']))

        return cf_pos 

    def import_cam_calib(self, calib_path):
        extr_calib_path = calib_path / "extrinsic_calib.json" 
        intr_calib_path = calib_path / "intrinsic_calib.json" 

        with open(extr_calib_path) as f:
            extrinsic_calib = json.load(f)
            rvec = np.array(extrinsic_calib['rvec'])
            tvec = np.array(extrinsic_calib['tvec'])

        with open(intr_calib_path) as f:
            intrinsic_calib = json.load(f)
            cam_mtx = np.array(intrinsic_calib['cam_mtx'])
            dist_coeffs = np.array(intrinsic_calib['dist_coeffs'])

        return cam_mtx, dist_coeffs, rvec, tvec

    def get_video_startime(self, video_file):
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
        self.frame_id = frame_id
        self.update()

    def set_offset(self, offset):
        self.t_offset = (offset / int(self.fps)) - 10
        self.update()

    def update(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)  # Get current frame
        ret, frame = self.cap.read()

        trajectories = self.get_trajectories()
        # Plot trajectories over frame
        out_frame = self.plot(frame, trajectories, self.frame_id, self.trail_len)
        # Display the resulting frame
        cv2.imshow(self.window_name, out_frame)

    def get_trajectories(self):
        t_frames_shifted = self.t_frames + self.t_offset
        trajectories = []
        # Plot trajectory of each drone
        for data in self.cf_pos:
            t, pos = data[:, 0], data[:, 1:]
            pos_interp = np.array([np.interp(t_frames_shifted, t, pos[:, i]) for i in range(pos.shape[1])]).T
            img_points, _ = cv2.projectPoints(pos_interp, self.rvec, self.tvec, self.cam_mtx, self.dist_coeffs)
            img_points = img_points.squeeze()
            trajectories.append(img_points)

        return trajectories

    def plot(self, img, trajectories, frame_id, trail_len):
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
            # self.ax.plot(traj[plot_range, 0], traj[plot_range, 1], linestyle='-')

        # Retrieve view of render buffer. See: https://stackoverflow.com/a/62040123
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        out_frame = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGB2BGR)

        return out_frame

    def run(self):
        self.update()  # Initialize screen with first frame
        while cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(500)

            if keyCode == ord('k'):
                break

        cv2.destroyAllWindows()
    
    def save(self):
        video_out_path = self.dir_path / "output.mp4"
        out = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

        pbar = tqdm(total=self.n_frames)  # Progress bar

        frame_id = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # Reset to start
        trajectories = self.get_trajectories()

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            # if frame_id > 200:
            #     break
            
            out_frame = self.plot(frame, trajectories, frame_id, self.trail_len)

            # Display the resulting frame
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
    args = parser.parse_args()

    video_file = args.video_file
    log_file = args.log_file

    plotter = Plotter(video_file, log_file, trail_len=30)
    plotter.run()
    plotter.save()