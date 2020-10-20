from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
import argparse
from datetime import datetime
from pathlib import Path


def create_checkerboard_3d(pattern_size, square_size):
    # World coordinates of corners on checkerboard
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    return pattern_points

def get_calibration_points(video_path, pattern_points, auto=False, n_calib_imgs=30):
    # Creating vector to store vectors of 3D points for each checkerboard image
    obj_points = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    img_points = [] 

    cap = cv2.VideoCapture(str(video_path))

    # Check if camera opened successfully
    if not cap.isOpened(): 
        raise FileNotFoundError(f"Video file {video_path} does not exist.")

    print(f"Please select {n_calib_imgs} images for the intrinsic calibration.")
    print("'q': quit\n'y': use image for calibration\n'any other key': skip image")

    winname = 'Calibration image'
    cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 960, 540)

    frame_id = 0
    skip = 5
    ret = True
    while cap.isOpened() and ret:
        ret, frame = cap.read()

        # Read every skip'th frame to avoid too many similar images
        if ret and frame_id % skip == 0 and len(obj_points) < n_calib_imgs:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cv2.cornerSubPix need gray scale image

            found, corners = cv2.findChessboardCorners(img_gray, pattern_size)

            if found:
                # Refining pixel coordinates for given 2d points
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)

                if auto:
                    obj_points.append(pattern_points)
                    img_points.append(corners_refined)
                else:
                    # Draw image with chessboard corners for manual inspection and let user decide
                    # if corner localization quality is good.
                    img = cv2.drawChessboardCorners(frame, pattern_size, corners_refined, found)
                    cv2.imshow(winname, img)
                    key_code = cv2.waitKey(0)
                    
                    if key_code == ord('y'):
                        obj_points.append(pattern_points)
                        img_points.append(corners_refined)
                    elif key_code == ord('q'):
                        break
        frame_id += 1
    cap.release()
    cv2.destroyAllWindows()

    if len(img_points) < n_calib_imgs:
        print(f"WARNING: Less than {n_calib_imgs} images for calibration.")
    else:
        print(f"Using {n_calib_imgs} images for calibration.\n")

    h, w = img_gray.shape[:2]  # Image size in px
    return obj_points, img_points, (w, h)

def calibrate(obj_points, img_points, img_size_wh):
    cam_mtx_guess = np.eye(3, 3)
    dist_coeffs_guess = np.zeros(5)

    flags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO
    flags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_RATIONAL_MODEL
    flags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST
    rms, cam_mtx, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points,
                                                                    img_points, 
                                                                    img_size_wh,
                                                                    cam_mtx_guess,
                                                                    dist_coeffs_guess,
                                                                    flags=flags)
    print(f"Average reprojection error: {rms}")
    return cam_mtx, dist_coeffs

def print_intrinsic_parameters(cam_mtx, dist_coeffs, img_size_wh):
    w, h = img_size_wh

    print("\nCamera parameters:")
    print(f"F_x: {cam_mtx[0, 0]:>10.2f} px at image size {w} x {h}")
    print(f"F_y: {cam_mtx[1, 1]:>10.2f} px at image size {w} x {h}")
    print(f"c_x: {cam_mtx[0, 2]:>10.2f} px, (ideal x center {w / 2} px)")
    print(f"c_y: {cam_mtx[1, 2]:>10.2f} px, (ideal y center {h / 2} px)")
    print(f"dist_coeffs: {dist_coeffs}\n")

def save_intrinsic_calib(output_dir, cam_mtx, dist_coeffs, img_size_wh):
    """Save json file with intrinsic calibration in output directory."""
    intrinsic_calib_data = {'img_size_wh': img_size_wh, 'cam_mtx': cam_mtx.tolist(), 'dist_coeffs': dist_coeffs.tolist()}

    file_path = output_dir / 'intrinsic_calib.json'

    if file_path.exists():
        answer = None
        while answer not in ['y', 'n']:
            answer = input("Intrinsic calibration file already exists. Do you want to overwrite? (y/n): ")
            if answer == 'y':
                break
            elif answer == 'n':
                time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                file_name = f"intrinsic_calib_{time_str}.json"
                file_path = output_dir / file_name
                break

    with open(file_path, "w") as f:
        json.dump(intrinsic_calib_data, f)
    print(f"Saved intrinsic calibration at {file_path}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate intrinsic camera calibration from video file')
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('--output', type=str, help='Output directory for calibration file')
    parser.add_argument('--columns', default=9, type=int, help='Pattern columns')
    parser.add_argument('--rows', default=6, type=int, help='Pattern rows')
    parser.add_argument('--square_size', default=25.0, type=float, help='Size of squares in pattern in mm')
    parser.add_argument('--auto', action='store_true', help='Automatically select images for calibration')
    args = parser.parse_args()

    # Define checkerboard
    pattern_size = (args.rows, args.columns)  # Number of internal checkerboard corners
    square_size = args.square_size / 1000  # Size of checkerboard square in meters

    # Path to calibration video
    video_path = Path(args.video_file)
    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = video_path.parent
        
    pattern_points = create_checkerboard_3d(pattern_size, square_size)
    obj_points, img_points, img_size_wh = get_calibration_points(video_path,
                                                                 pattern_points,
                                                                 auto=args.auto,
                                                                 n_calib_imgs=30)
    cam_mtx, dist_coeffs = calibrate(obj_points, img_points, img_size_wh)
    print_intrinsic_parameters(cam_mtx, dist_coeffs, img_size_wh)
    save_intrinsic_calib(output_dir, cam_mtx, dist_coeffs, img_size_wh)