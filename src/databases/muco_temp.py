import os

import cv2

from util.misc import load

MUCO_TEMP_PATH = '../datasets/MucoTemp'


def get_frame(cam, vid_id, frame_ind, rgb=True):
    path = os.path.join(MUCO_TEMP_PATH, 'frames/cam_%d/vid_%d' % (cam, vid_id), 'img_%04d.jpg' % frame_ind)
    img = cv2.imread(path)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_metadata():
    """ Returns metadata for each video in the video. It contains the original videos and starting frames each video contains. """
    return load(os.path.join(MUCO_TEMP_PATH, 'sequence_meta.pkl'))


def load_gt(cam):
    return load(os.path.join(MUCO_TEMP_PATH, 'frames', 'cam_%d' % cam, 'gt.pkl'))


def load_hrnet(cam, vid):
    return load(os.path.join(MUCO_TEMP_PATH, 'hrnet_keypoints', 'cam_%d' % cam, 'gt_match_posedist_80', 'vid_%d.pkl' % vid))
