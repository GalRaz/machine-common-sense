from datetime import datetime
import pandas as pd
from string import digits
import json
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

# jspych dir
JSPSYCH_DIR = "/nese/mit/group/saxelab/users/galraz/pkbb/data/exp3/"

# point to folder with corresponding video
VIDEO_DIR = "/nese/mit/group/saxelab/users/galraz/pkbb/videos/exp3/"

# attention getter frame
ATTENTION_GETTER_IMAGE = "/nese/mit/group/saxelab/users/galraz/icatcher_tests/machine-common-sense/parsers/attngetter.png" 
AG_small = "/nese/mit/group/saxelab/users/galraz/icatcher_tests/machine-common-sense/parsers/attngetter_small.png" 


# histogram matching parameters
histSize = [40, 40]

# hue varies from 0 to 179, saturation from 0 to 255
h_ranges = [0, 256]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
# Use the 0-th and 1-st channels
channels = [0, 1]

# max chi square value for considering frame and attngetter a match
hist_comp_thresh = 0.6
black_frame_threshold = 0.95

# how many frames to skip at beginning
start_frame_number = 1500


# get file names (exclude hidden files and folders)
filenames = [f for f in os.listdir(JSPSYCH_DIR) if not f.startswith(".") and f.endswith(".csv")]

child_ids = [file.replace("_jspsych.csv", "") for file in filenames]

# attention getter frames
attngetter_frames = dict()

for child in child_ids:
    
    video_path = VIDEO_DIR + child + "_SS.mp4"
    
    # get attn getter       
    attngetter = cv.imread(ATTENTION_GETTER_IMAGE)
    # hsv_attngetter = cv.cvtColor(attngetter, cv.COLOR_BGR2HSV)
    hist_attngetter = cv.calcHist([attngetter], channels, None, histSize, ranges, accumulate=False).flatten()
    
    # normalize
    hist_attngetter = hist_attngetter / np.sum(hist_attngetter)

    ag_small = cv.imread(AG_small)
    hist_ag_small = cv.calcHist([ag_small], channels, None, histSize, ranges, accumulate=False).flatten()
    hist_ag_small = hist_ag_small / np.sum(hist_ag_small)
    
    # get video
    cap = cv.VideoCapture(video_path)
       
    # skip first n frames
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame_number)

    # get first frame of video
    success, img = cap.read()
    
    fno = start_frame_number;

    while success:
        
        # sample every 2nd frame
        if fno % 2 == 0:
            
            _, img = cap.retrieve()

            print('frame num: ', fno)
            
            # calculate histogram match
            # hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hist_img = cv.calcHist([img], channels, None, histSize, ranges, accumulate=False).flatten()
            
            hist_img = hist_img / np.sum(hist_img)
            frame_corr = cv.compareHist(hist_attngetter, hist_img, cv.HISTCMP_CORREL)
            
            # attention getter has been found it hist meets threshold and frame is not completely black         
            if (frame_corr > hist_comp_thresh) & (hist_img[0] < black_frame_threshold):
                a = 0;
                # frame_no, timestamp = get_info(img)
                current_fno = int(cap.get(1))
                fps = cap.get(cv.CAP_PROP_FPS)
                attngetter_frames[child] = [current_fno, current_fno/fps]
                print('found attention getter')
                break;
            
        # read next frame
        success = cap.grab()
        fno += 1
    
    # load jspsych files to compute relative timestamps
    jspsych_file = pd.read_csv(JSPSYCH_DIR)
    
    
