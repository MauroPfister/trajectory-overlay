from matplotlib import pyplot as plt
import argparse
import matplotlib as mpl
import numpy as np
import cv2
import json
from pathlib import Path


def draw_marker(x, y, img, color=(0, 0, 255), cross_size=5):
    x, y = int(x), int(y)
    cv2.line(img, (x - cross_size, y), (x + cross_size, y), color, thickness=1)
    cv2.line(img, (x, y - cross_size), (x, y + cross_size), color, thickness=1)
    cv2.circle(img, (x, y), 3, color, 1)

def click_event(event, x, y, flags, params): 
  
    window_name = params[0]
    img = params[1]
    calibration_markers_img = params[2]

    # Checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        calibration_markers_img.append([x, y]) 
        draw_marker(x, y, img, color=(0, 0, 255))  # Paint marker location red
        cv2.imshow(window_name, img) 

def get_marker_img_pos(img):
    """Let user draw calibration marker positions in image"""
    window_name = 'image'
    marker_img_pos = []

    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.imshow(window_name, img) 

    # Setting mouse handler for the image and calling the click_event() function 
    cv2.setMouseCallback(window_name, click_event, [window_name, img, marker_img_pos]) 

    # Wait for a key to be pressed to exit 
    print("Please select 6 markers with the right mouse button, then press any key.")
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    return np.array(marker_img_pos).astype(np.float32)

def calculate_cam_pose(marker_pos, marker_img_pos, cam_mtx, dist_coeffs):
    """Calculate camera pose using marker positions in 3D and 2D."""
    rms, rvec, tvec = cv2.solvePnP(marker_pos, marker_img_pos, cam_mtx, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

    return rvec, tvec

def check_reprojection(img, marker_pos, marker_img_pos, rvec, tvec, cam_mtx, dist_coeffs):
    marker_img_pos_proj, _ = cv2.projectPoints(marker_pos, rvec, tvec, cam_mtx, dist_coeffs)
    marker_img_pos_proj = marker_img_pos_proj.squeeze()

    # Check accuracy by reprojecting calibration markers
    error = np.linalg.norm(marker_img_pos - marker_img_pos_proj) / len(marker_img_pos)
    print(f"RMS error: {error}")

    for p in marker_img_pos:
        draw_marker(*p, img, color=(0, 0, 255))
    for p in marker_img_pos_proj:
        draw_marker(*p, img, color=(0, 255, 0))

    # Draw world axis
    cv2.drawFrameAxes(img, cam_mtx, dist_coeffs, rvec, tvec, 1)
    
    win_name = 'Reprojection'
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)
    cv2.imshow(win_name, img)
    print("Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_extrinsic_calib(output_dir, rvec, tvec):
    """Save json file with extrinsic calibration in output directory."""
    extrinsic_calib_data = {"rvec": rvec.tolist(), "tvec": tvec.tolist()}

    file_path = output_dir / 'extrinsic_calib.json'

    if file_path.exists():
        answer = None
        while answer not in ['y', 'n']:
            answer = input("Extrinsic calibration file already exists. Do you want to overwrite? (y/n): ")
            if answer == 'y':
                break
            elif answer == 'n':
                time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                file_name = f"extrinsic_calib_{time_str}.json"
                file_path = output_dir / file_name
                break

    with open(file_path, "w") as f:
        json.dump(extrinsic_calib_data, f)
    print(f"Saved extrinsic calibration at {file_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate extrinsic camera calibration')
    parser.add_argument('video_file', type=str, help="Path to the video file")
    parser.add_argument('marker_file', type=str, help="Path to the .json file containing the 3D positions of the markers")
    parser.add_argument('intrinsic_calib', type=str, help="Path to the .json file containing the intrinsic camera calibration")
    parser.add_argument('--output', type=str, help='Output directory for calibration file')
    args = parser.parse_args()

    # Path to calibration video
    video_path = Path(args.video_file)
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = video_path.parent

    # Get marker positions in 3D
    with open(args.marker_file) as f:
        marker_dict = json.load(f)
        marker_pos_tmp = np.array(marker_dict['pos'])
        marker_pos = np.array([marker_pos_tmp[:, 0], -marker_pos_tmp[:, 2], marker_pos_tmp[:, 1]]).T.astype(np.float32)

    with open(args.intrinsic_calib) as f:
        intrinsic_calib = json.load(f)
        cam_mtx = np.array(intrinsic_calib['cam_mtx'])
        dist_coeffs = np.array(intrinsic_calib['dist_coeffs'])

    # Get first available frame of video
    cap = cv2.VideoCapture(args.video_file)
    ret = False
    while not ret:
        ret, frame = cap.read()

    # Get marker positions in 2D
    marker_img_pos = get_marker_img_pos(frame)

    rvec, tvec = calculate_cam_pose(marker_pos, marker_img_pos, cam_mtx, dist_coeffs)
    check_reprojection(frame, marker_pos, marker_img_pos, rvec, tvec, cam_mtx, dist_coeffs)
    save_extrinsic_calib(output_dir, tvec, rvec)
