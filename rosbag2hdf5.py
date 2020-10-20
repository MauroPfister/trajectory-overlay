
from datetime import datetime
import os
import argparse
import rosbag
import h5py
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rosbag to hdf5 converter')
    parser.add_argument('bagfile', type=str, help="Path to the .bag file")
    args = parser.parse_args()

    bagfile = args.bagfile

    bagfile_name, _ = os.path.splitext(bagfile)

    h5file = bagfile_name + '.h5'

    hf = h5py.File(h5file, 'w')

    print('reading rosbag ' + bagfile)
    bag = rosbag.Bag(bagfile, 'r')

    pos_data = {}

    for topic, msg, t in bag.read_messages(topics=['/tf']):

        if topic == '/tf':
            t_secs = msg.transforms[0].header.stamp.secs
            t_nsecs = msg.transforms[0].header.stamp.nsecs
            t = t_secs + t_nsecs / 1e9  # Float time in seconds
            # print(datetime.fromtimestamp(t).strftime("%A, %B %d, %Y %I:%M:%S.%f"))

            cf_id = msg.transforms[0].child_frame_id
            x = msg.transforms[0].transform.translation.x
            y = msg.transforms[0].transform.translation.y
            z = msg.transforms[0].transform.translation.z

            if cf_id not in pos_data:
                pos_data[cf_id] = [[t, x, y, z]]
            pos_data[cf_id].append([t, x, y, z]) 
            # print(msg.transforms[0].header)

    # Write to hdf5 file
    for cf_id, pos in pos_data.items():
        pos_np = np.array(pos)
        hf.create_dataset(cf_id + '/pos', data=pos_np)

    hf.close()