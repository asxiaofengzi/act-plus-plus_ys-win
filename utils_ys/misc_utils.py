import os
import re
import h5py
import cv2
import numpy as np
import csv
from utils_not_ys.train_utils import *
def load_hdf5_episode(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

# assumes that key of img_dict is camera from which image was recorded
def img_dict_to_video(img_dict, dt, video_path=None):
    cam_names = list(img_dict.keys())
    cam_names = sorted(cam_names)
    all_cam_videos = []
    for cam_name in cam_names:
        all_cam_videos.append(img_dict[cam_name])
    all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension
    n_frames, h, w, _ = all_cam_videos.shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for t in range(n_frames):
        image = all_cam_videos[t]
        image = image[:, :, [2, 1, 0]]  # swap B and R channel
        out.write(image)
    out.release()
    print(f'Saved video to: {video_path}')

def append_to_csv(filename,x_list):
    f = open(filename, mode='a', newline='')
    writer = csv.writer(f)
    writer.writerow(x_list)
    f.close()

def add_to_column_from_row(df, start_row, value):
    """
    Adds a given value to each element in the first column of the dataframe starting from a specific row.

    Parameters:
    df (pd.DataFrame): The dataframe to modify.
    start_row (int): The starting row index from where to begin addition.
    value (float or int): The value to add to each element in the first column.

    Returns:
    pd.DataFrame: The modified dataframe with the value added to the first column from the specified row.
    """
    # Ensure the start_row is within the bounds of the dataframe
    if start_row < 0 or start_row >= len(df):
        raise ValueError("start_row must be within the range of the dataframe's indices.")
    df.iloc[start_row:, 0] += value

    return df

def find_largest_suffix(directory, base_filename, extension):
    """
    Finds the largest numeric suffix in filenames within a given directory,
    based on a specified base filename and extension.

    Parameters:
    directory (str): The path to the directory containing the files.
    base_filename (str): The base name of the files to search for.
    extension (str): The file extension of the files to search for.

    Returns:
    int: The largest numeric suffix found in the filenames. Returns -1 if no valid files are found.
    """
    max_suffix = -1  # Initialize with -1 to indicate no valid files found initially

    # Construct a regular expression pattern dynamically based on base_filename and extension
    pattern = re.compile(rf'{re.escape(base_filename)}_(\d+){re.escape(extension)}$')

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the numeric part (group 1) from the match and convert to integer
            num = int(match.group(1))
            # Update max_suffix if this number is larger
            max_suffix = max(max_suffix, num)

    return max_suffix

def determine_logfiles_paths(logs_dir):
    if not logs_dir.endswith(('/','\\')):
        logs_dir=logs_dir+'/'
    validation_base_filename="validation_perf"
    eval_base_filename="eval_perf"
    extension='.csv'
    logfiles_path={}
    max_validation_suffix=find_largest_suffix(logs_dir, validation_base_filename, extension)
    logfiles_path['validation_file_path']=(logs_dir+validation_base_filename+"_"+
    str(max_validation_suffix+1)+extension)
    max_eval_suffix=find_largest_suffix(logs_dir, validation_base_filename, extension)
    logfiles_path['eval_file_path']=(logs_dir+eval_base_filename+"_"+
    str(max_eval_suffix+1)+extension)
    return logfiles_path

