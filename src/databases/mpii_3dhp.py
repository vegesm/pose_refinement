import os

import cv2
import numpy as np

from databases.joint_sets import MuPoTSJoints
from util.misc import load

MPII_3DHP_PATH = '../datasets/Mpi3DHP'


def test_frames(seq):
    frames = sorted(os.listdir(os.path.join(MPII_3DHP_PATH, 'mpi_inf_3dhp_test_set', 'TS%d' % seq, 'imageSequence')))

    # In TS3/TS4 last (two) frames do not have annotations
    if seq == 3:
        frames = frames[:-1]
    elif seq == 4:
        frames = frames[:-2]

    return frames


def num_test_frames(seq):
    num_frames = len(os.listdir(os.path.join(MPII_3DHP_PATH, 'mpi_inf_3dhp_test_set', 'TS%d' % seq, 'imageSequence')))

    # In TS3/TS4 last (two) frames do not have annotations
    if seq == 3:
        num_frames -= 1
    elif seq == 4:
        num_frames -= 2

    return num_frames


def get_test_image(seq, frame_ind, rgb=True):
    """ frame_ind is indexed from 0, while filename is indexed from 1!!"""
    img = cv2.imread(os.path.join(MPII_3DHP_PATH, 'mpi_inf_3dhp_test_set',
                                  'TS%d' % seq, 'imageSequence', 'img_%06d.jpg' % (frame_ind + 1)))
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_image(subject, sequence, camera, frame, rgb=True):
    img = cv2.imread(os.path.join(MPII_3DHP_PATH, 'frames', 'S%d' % subject, 'Seq%d' % sequence, 'imageSequence',
                                  "img_%d_%06d.jpg" % (camera, frame + 1)))
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_mask(subject, sequence, camera, frame, mask_type):
    assert mask_type in ['ChairMasks', 'FGmasks'], "unknown mask type: " + mask_type
    img = cv2.imread(
        os.path.join(MPII_3DHP_PATH, 'frames', 'S%d' % subject, 'Seq%d' % sequence, mask_type, "img_%d_%06d.jpg" % (camera, frame + 1)))

    return img


def get_train_fps(sub, seq):
    assert 1 <= sub <= 8
    assert seq in (1, 2)

    if sub == 3 or sub == 5 or (sub == 1 and seq == 2):
        return 50
    else:
        return 25


def test_poses_hrnet(seq):
    return load(os.path.join(MPII_3DHP_PATH, 'mpi_inf_3dhp_test_set', 'TS%d' % seq, 'hrnet.pkl'))


def train_poses_hrnet(sub, seq, cam):
    return load(os.path.join(MPII_3DHP_PATH, 'S%d' % sub, 'Seq%d' % seq, 'hrnet', 'hrnet_%02d.pkl' % cam))


# filters out the relevant 17 joints from the raw annot.mat files. Based on mpii_get_joint_set.m
MUPOTS_RELEVANT_JOINTS = np.array([8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]) - 1


def train_ground_truth(sub, seq, fix_incorrect=True):
    """
    Returns the ground truth annotations. Returns a dict with fields 'annot2', 'annot3', 'univ_annot3'
    :param fix_incorrect: S4/Seq2 has annotations flipped on some frames, if True they are flipped back
    :return:
    """
    annot = load(os.path.join(MPII_3DHP_PATH, 'S%d' % sub, 'Seq%d' % seq, 'annot.mat'))
    annot2 = list([x[0].reshape((-1, 28, 2))[:, MUPOTS_RELEVANT_JOINTS].astype('float32') for x in annot['annot2']])
    annot3 = list([x[0].reshape((-1, 28, 3))[:, MUPOTS_RELEVANT_JOINTS].astype('float32') for x in annot['annot3']])
    univ_annot3 = list([x[0].reshape((-1, 28, 3))[:, MUPOTS_RELEVANT_JOINTS].astype('float32') for x in annot['univ_annot3']])
    assert np.all(annot['cameras'][0] == np.arange(14))
    assert np.all(annot['frames'][:, 0] == np.arange(len(annot2[0])))

    # S3/Seq1 has one extra annotation but one less frame
    # Remove the very last annotation from everywhere
    if sub == 3 and seq == 1:
        for cam in range(14):
            annot2[cam] = annot2[cam][:-1]
            annot3[cam] = annot3[cam][:-1]
            univ_annot3[cam] = univ_annot3[cam][:-1]

    if sub == 4 and seq == 2 and fix_incorrect:
        # between 3759(in) and 5853(ex) annotations are flipped
        for cam in range(14):
            annot2[cam][3759:5853] = MuPoTSJoints().flip(annot2[cam][3759:5853])
            annot3[cam][3759:5853] = MuPoTSJoints().flip(annot3[cam][3759:5853])
            univ_annot3[cam][3759:5853] = MuPoTSJoints().flip(univ_annot3[cam][3759:5853])

    N = len(annot2[0])
    for cam in range(14):
        assert len(annot2[cam]) == N
        assert len(annot3[cam]) == N
        assert len(univ_annot3[cam]) == N

    result = {'annot2': annot2, 'annot3': annot3, 'univ_annot3': univ_annot3}

    return result


def get_test_calib(seq):
    assert 1 <= seq <= 6, seq

    # Numbers are coming from the "mpi_inf_3dhp_test_set/test_util/camera_calibration/*.calib" files
    if 1 <= seq <= 4:
        f = 7.32506 / 10 * 2048
        cx = 1024 - 0.0322884 / 10 * 2048
        cy = 1024 + 0.0929296 / 10 * 2048
    elif 5 <= seq <= 6:
        f = 8.770747185 / 10 * 1920
        cx = 1920 / 2 - 0.104908645 / 10 * 1920
        cy = 1080 / 2 + 0.104899704 / 5.625000000 * 1080

    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype='float32')


def get_calibration_matrices():
    """

    Returns:
        dict (subject, seq, camera) to intrinsic camera matrix
    """
    calibs = {}
    for subject in range(1, 9):
        for seq in [1, 2]:
            with open(MPII_3DHP_PATH + '/S%d/Seq%d/camera.calibration' % (subject, seq)) as f:
                data = f.readlines()
            data = [x.strip() for x in data]
            camera = None
            for line in data:
                if line.startswith("name"):
                    camera = int(line[5:])
                elif line.startswith("intrinsic"):
                    assert camera is not None
                    #                 print line
                    line = line[len("intrinsic"):].strip()
                    parts = line.split(' ')
                    parts = list(map(float, parts))
                    assert len(parts) == 16

                    c = np.eye(3, dtype='float32')
                    c[0, 0] = parts[0]
                    c[0, 2] = parts[2]
                    c[1, 1] = parts[5]
                    c[1, 2] = parts[6]
                    calibs[(subject, seq, int(camera))] = c

    return calibs
