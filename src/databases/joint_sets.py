import numpy as np
from util.misc import assert_shape

# SIDEDNESS
# 0 - right
# 1 - left
# 2 - center

class JointSet:

    def index_of(self, joint_name):
        joint_inds = np.where(self.NAMES == joint_name)[0]
        assert len(joint_inds) > 0, "No joint called " + joint_name
        return joint_inds[0]

    def flip(self, data):
        """ Flips a dataset """
        assert_shape(data, ('*', self.NUM_JOINTS, None))

        data = data.copy()
        data[...,self.JOINTS_LEFT+self.JOINTS_RIGHT,:] = data[...,self.JOINTS_RIGHT+self.JOINTS_LEFT,:]
        return data


class MuPoTSJoints(JointSet):
    NAMES = np.array(["head_top", 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',  # 0-4
                      'left_shoulder', 'left_elbow', 'left_wrist',  # 5-7
                      'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',  # 8-13
                      'hip', 'spine', 'head/nose'])
    NUM_JOINTS = 17
    TO_COMMON14 = [14, 8, 9, 10, 11, 12, 13, 1, 5, 6, 7, 2, 3, 4]

    LIMBGRAPH = [(10, 9), (9, 8), (8, 14),  # rleg
                 (13, 12), (12, 11), (11, 14),  # llel
                 (0, 16), (16, 1),  # head to thorax
                 (1, 15), (15, 14),  # thorax to hip
                 (4, 3), (3, 2), (2, 1),  # rarm
                 (7, 6), (6, 5), (5, 1),  # larm
                 ]

    SIDEDNESS = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1]
    JOINTS_LEFT = [5, 6, 7, 11, 12, 13]
    JOINTS_RIGHT = [2, 3, 4, 8, 9, 10]
    NAMES.flags.writeable = False


class OpenPoseJoints(JointSet):
    NAMES = np.array(['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder', 'left_elbow',
                      'left_wrist', 'hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle',
                      'right_eye', 'left_eye', 'right_ear', 'left_ear',
                      'left_bigtoe', 'left_smalltoe', 'left_heel', 'right_bigtoe', 'right_smalltoe', 'right_heel'])

    NUM_JOINTS = 25
    TO_COMMON14 = [8, 9, 10, 11, 12, 13, 14,  # hip, rleg, lleg
                   1, 5, 6, 7, 2, 3, 4]  # neck/thorax, larm, rarm

    STABLEJOINTS = np.arange(17)

    NAMES.flags.writeable = False


class PanopticJoints(JointSet):
    NAMES = np.array(['neck', 'nose', 'hip',
                      'left_shoulder', 'left_elbow', 'left_wrist',  # 3-5
                      'left_hip', 'left_knee', 'left_ankle',  # 6-8
                      'right_shoulder', 'right_elbow', 'right_wrist',  # 9-11
                      'right_hip', 'right_knee', 'right_ankle',  # 12-14
                      'left_eye', 'left_ear', 'right_eye', 'right_ear'])

    NUM_JOINTS = 19
    TO_COMMON14 = [2, 12, 13, 14, 6, 7, 8, 0, 3, 4, 5, 9, 10, 11]  # neck/thorax, larm, rarm
    LIMBGRAPH = [(0, 1), (0, 2),  # spine
                 (0, 3), (3, 4), (4, 5),  # larm
                 (2, 6), (6, 7), (7, 8),  # lleg
                 (0, 9), (9, 10), (10, 11),  # rarm
                 (2, 12), (12, 13), (13, 14),  # rleg
                 (1, 15), (15, 16),  # lface
                 (1, 17), (17, 18)  # rface
                 ]
    SIDEDNESS = [2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

    NAMES.flags.writeable = False


class CocoExJoints(JointSet):
    NAMES = np.array(["nose", "left_eye", "right_eye", "left_ear", "right_ear",  # 0-4
                      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",  # 5-10
                      "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",  # 11-16
                      "hip", "neck"])
    NUM_JOINTS = 19
    TO_COMMON14 = [17, 12, 14, 16, 11, 13, 15, 18, 5, 7, 9, 6, 8, 10]
    LIMBGRAPH = [(0, 1), (1, 3),  # lface
                 (0, 2), (2, 4),  # rface
                 (0, 18), (18, 17),  # spine
                 (18, 5), (5, 7), (7, 9),  # larm
                 (18, 6), (6, 8), (8, 10),  # rarm
                 (17, 11), (11, 13), (13, 15),  # lleg
                 (17, 12), (12, 14), (14, 16)]  # rleg
    SIDEDNESS = [1, 1, 0, 0, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]

    JOINTS_LEFT = [1, 3, 5, 7, 9, 11, 13, 15]
    JOINTS_RIGHT = [2, 4, 6, 8, 10, 12, 14, 16]

    NAMES.flags.writeable = False


class Common14Joints(JointSet):
    NAMES = np.array(['hip', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'neck',
                      'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'])
    NUM_JOINTS = 14
    TO_COMMON14 = np.arange(14)

    LIMBGRAPH = [(0, 1), (1, 2), (2, 3),  # rleg
                 (0, 4), (4, 5), (5, 6),  # lleg
                 (0, 7),  # spine
                 (7, 8), (8, 9), (9, 10),  # larm
                 (7, 11), (11, 12), (12, 13)]  # rarm
    SIDEDNESS = [0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0]

    NAMES.flags.writeable = False
