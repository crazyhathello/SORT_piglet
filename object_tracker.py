from __future__ import print_function

import argparse
import os.path
import time
from sort import Sort
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvas
import imageio
from collections import deque
import cv2
import math
from matplotlib import lines
from tqdm import tqdm

np.random.seed(0)

def main():
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32,3)
    

    box_type = "horizontal"
    mot_tracker = Sort() #create instance of the SORT tracker
    if not os.path.exists('Result/{}/Tracking {} result'.format(args.seq_path.split('/')[-1], box_type)):
        os.makedirs('Result/{}/Tracking {} result'.format(args.seq_path.split('/')[-1], box_type))
        
    total_time, total_frames, total_frames = tracking(total_time, total_frames, colours, mot_tracker, box_type)
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

def tracking(total_time, total_frames, colours, mot_tracker, box_type):
    seq_dets = np.loadtxt("{}/{}_boxes.csv".format(args.seq_path, box_type), delimiter=',')
    images = []
    with open('Result/{}/Tracking {} result/tracking_result.txt'.format(args.seq_path.split('/')[-1], box_type), "w") as out_file:
        for frame in tqdm(np.unique(seq_dets[:,0]).astype(np.int32)):
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            total_frames += 1
            start_time = time.time()
            trackers = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            for d in trackers:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)

    return total_time, total_frames, total_frames



if __name__ == '__main__':
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument('--save_fig', dest='save_fig', help='Save image', action='store_true')
    parser.add_argument('--write_video', dest='write_video', help='write video', action='store_true')
    # csv path
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='../../data/test/images')
    parser.add_argument('--is_rotated', dest='is_rotated', help='is rotated', action='store_true')
    
    args = parser.parse_args()
    main()