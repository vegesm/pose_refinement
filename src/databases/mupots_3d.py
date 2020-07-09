import glob
import os

import cv2
import numpy as np

from databases.joint_sets import MuPoTSJoints
from util.misc import load, assert_shape
from util.mx_tools import calibration_matrix

MUPO_TS_PATH = '../datasets/MuPoTS'


def _decode_sequence(sequence):
    assert isinstance(sequence, (int, np.int32, str)), "sequence must be an int or string"

    if isinstance(sequence, (int, np.int32)):
        assert 1 <= sequence <= 20, "sequence id must be between 1 and 20"
        sequence = "TS" + str(sequence)

    return sequence


def get_frame_files(sequence):
    """
    Returns the list of jpg files for a given video sequence.

    Parameters:
        sequence: either an int between 1 and 20 or a string in the form TSx.
    """
    sequence = _decode_sequence(sequence)

    folder = os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence)
    assert os.path.isdir(folder), "Could not find " + folder

    return sorted(glob.glob(folder + '/*.jpg'))


def _concat_raw_gt(gt, field, dtype):
    """ Concatenates gt annotations coming from the annot.mat file. """
    data = np.empty(gt.shape + gt[0, 0][field][0, 0].T.shape, dtype=dtype)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = gt[i, j][field][0, 0].T

    return data


def load_gt_annotations(sequence):
    """
    Loads GT annotations as numpy arrays. This method cleans up the unnecessary indices
    resulting from the `.mat` file loading.
    Returns a dict. Has the following keys:

    - annot2: (nFrames, nPoses, 17, 2), float32
    - annot3: (nFrames, nPoses, 17, 3), float32, the unnormalized coordinates
    - univ_annot3: (nFrames, nPoses, 17, 3), float32
    - isValidFrame: (nFrames, nPoses), bool
    - occlusions: (nFrames, nPoses, 17), bool
    """
    data = load_raw_gt_annotations(sequence)
    occlusions = load_raw_gt_occlusions(sequence)

    occ_out = np.empty(occlusions.shape + (17,), dtype='bool')
    for i in range(occlusions.shape[0]):
        for j in range(occlusions.shape[1]):
            occ_out[i, j] = occlusions[i, j][0]

    result = {'annot2': _concat_raw_gt(data, 'annot2', 'float32'),
              'annot3': _concat_raw_gt(data, 'annot3', 'float32'),
              'univ_annot3': _concat_raw_gt(data, 'univ_annot3', 'float32'),
              'isValidFrame': _concat_raw_gt(data, 'isValidFrame', 'bool').squeeze(),
              'occlusions': occ_out}

    return result


def load_raw_gt_annotations(sequence):
    """ Loads the GT annotations from the MuPo-TS `annnot.mat` file. """
    sequence = _decode_sequence(sequence)
    return load(os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence, 'annot.mat'))['annotations']


def load_raw_gt_occlusions(sequence):
    sequence = _decode_sequence(sequence)
    return load(os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence, 'occlusion.mat'))['occlusion_labels']


def load_2d_predictions(sequence, detector):
    """ Loads precreated 2D pose predictions. These are matched with GT poses. """
    assert detector == 'hrnet'
    sequence = _decode_sequence(sequence)

    return load(os.path.join(MUPO_TS_PATH, "hrnet_pose2d", sequence + '.pkl'), pkl_py2_comp=True)


def all_sequences():
    """ Returns every available sequence's name. """
    folder = os.path.join(MUPO_TS_PATH, "MultiPersonTestSet")
    assert os.path.isdir(folder), "Could not find " + folder

    return sorted(os.listdir(folder))


def _sequence2num(sequence):
    """ Returns the input sequence as a number. """
    if isinstance(sequence, str):
        sequence = int(sequence[2])

    return sequence


def get_fps(sequence):
    sequence = _sequence2num(sequence)
    return 30 if sequence <= 5 else 60


def get_image(sequence, frameind):
    """

    :param sequence: sequence id
    :param frameind: zero based index of the image
    :return:
    """
    sequence = _decode_sequence(sequence)
    img = cv2.imread(os.path.join(MUPO_TS_PATH, "MultiPersonTestSet", sequence, 'img_%06d.jpg' % frameind))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_size(sequence):
    """
    Returns:
        width, height
    """
    sequence = _sequence2num(sequence)
    return (2048, 2048) if sequence <= 5 else (1920, 1080)


def get_calibration_matrices():
    calibs = {}
    for seq in range(1, 21):
        annot = load_gt_annotations(seq)

        valid = np.logical_and(annot['isValidFrame'][:, :, np.newaxis], annot['occlusions'])

        pose2d = annot['annot2'][valid]
        pose3d = annot['annot3'][valid]

        calibs[seq], reproj, resx, resy, _, _ = calibration_matrix(pose2d, pose3d)

    return calibs


def _match_poses(gt_pose_2d, gt_visibility, pred_pose_2d, pred_visibility, threshold, verbose=False):
    """
    Implements ``mpii_multiperson_get_identity_matching.m``.

    Parameters:
        gt_pose_2d: (nGtPoses,nJoints,2), ground truth 2D poses on the image.
        gt_visibility: (nGtPoses,nJoints), True if the given joint is visible in ground-truth.
        gt_pose_2d: (nPredPoses,nJoints,2) predicted 2D poses on the image.
        pred_visibility: (nPredPoses,nJoints), True if the given joint is visible in predictions.

    Returns:
        ndarray(nGtPoses), the indices of the matched predicted pose for all ground truth poses. If no
        matches were found, the value is -1.
    """
    pair_ind = -np.ones(len(gt_pose_2d), dtype='int64')  # -1 means no pair, otherwise the pair id
    has_gt_pair = np.zeros(len(pred_pose_2d), dtype='bool')  # True means the predicted pose is already matched up

    if verbose:
        print(gt_visibility)
        print(pred_visibility)

    for i in range(len(gt_pose_2d)):
        diff = np.abs(gt_pose_2d[[i]] - pred_pose_2d)  # (nPredPose, nJoints, 2)
        matches = np.all(diff < threshold, axis=2)  # (nPredPose, nJoints)
        match_scores = np.sum(matches * (gt_visibility[[i]] & pred_visibility), axis=1)
        match_scores[has_gt_pair] = 0  # zero out scores for already matched up pred_poses

        if verbose:
            print(match_scores)

        best_match_ind = np.argmax(match_scores)
        if match_scores[best_match_ind] > 0:
            pair_ind[i] = best_match_ind
            has_gt_pair[best_match_ind] = True

    return pair_ind


# Parents of joints in MuPoTS joint set
_JOINT_PARENTS = np.array([2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2]) - 1
# The order in which joints are scaled, from the hip to outer limbs
_TRAVERSAL_ORDER = np.array([15, 16, 2, 1, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) - 1


def _scale_to_gt(pred_poses, gt_poses):
    """ Scales bone lengths in pred_poses to match gt_poses. Corresponds to ``mpii_map_to_gt_bone_lengths.m``."""
    assert_shape(pred_poses, (None, 17, 3))
    assert_shape(gt_poses, (None, 17, 3))

    rescaled_pred_poses = pred_poses.copy()

    for ind in _TRAVERSAL_ORDER:
        parent = _JOINT_PARENTS[ind]
        gt_bone_length = np.linalg.norm(gt_poses[:, ind] - gt_poses[:, parent], axis=1)  # (nPoses,)
        pred_bone = pred_poses[:, ind] - pred_poses[:, parent]  # (nPoses, 3)
        pred_bone = pred_bone * gt_bone_length[:, np.newaxis] / \
                    (np.linalg.norm(pred_bone, axis=1, keepdims=True) + 1e-8)
        rescaled_pred_poses[:, ind] = rescaled_pred_poses[:, parent] + pred_bone

    return rescaled_pred_poses


PCK_THRESHOLD = 150
AUC_THRESHOLDS = np.arange(0, 151, 5)


def eval_poses(matched_only, is_relative, pose3d_type, preds_2d_kpt, preds_3d_kpt, keep_matching=False):
    """
    Calculates the PCK and AUC. This function is equivalent to ``mpii_mupots_multiperson_eval.m``.
    It performs the same gt scaling transformation, uses the same joints for matching and evaluation.

    :param matched_only: True if only detected poses count towards the PCK and AUC
    :param is_relative: True if relative error is calculated
    :param pose3d_type: 'annot3' or 'univ_annot3'
    :param preds_2d_kpt: seq->list(ndarray(nPoses,17,2)), in MuPo-TS joint order. 2D pose predictions.
    :param preds_3d_kpt: seq->list(ndarray(nPoses,17,2)), in MuPo-TS joint order. 3D pose predictions.
    :param keep_matching: if True, the preds_2d_kpt arrays are assumed to be already matched with gt.
                          Otherwise, the matching algorithm in mpii_map_to_gt_bone_lengths is used.
    :return: two dicts from seq name to pck and auc
    """

    # Joints used in original evaluation script
    joints_for_matching = np.arange(1, 14)  # Joints used to match up the 2D poses
    joint_groups = [['Head', [0]], ['Neck', [1]], ['Shou', [2, 5]], ['Elbow', [3, 6]],
                    ['Wrist', [4, 7]], ['Hip', [8, 11]], ['Knee', [9, 12]], ['Ankle', [10, 13]]]
    scored_joints = np.concatenate([x[1] for x in joint_groups])  # Those joints that take part in scoring

    my_matching_inds = []
    all_perjoint_errors = {}
    pck_by_sequence = {}
    auc_by_sequence = {}
    for seq in range(1, 21):
        gt = load_gt_annotations(seq)
        num_frames = gt['annot2'].shape[0]

        gt_poses = []
        pred_poses = []
        valid_pred = []
        for i in range(num_frames):
            gt_pose_2d = gt['annot2'][i][gt['isValidFrame'][i]]
            gt_pose_3d = gt[pose3d_type][i][gt['isValidFrame'][i]]
            # gt_visibility = ~gt['occlusions'][i][gt['isValidFrame'][i]]
            gt_visibility = np.ones(gt_pose_2d.shape[:2], dtype='bool')

            pred_pose_2d = preds_2d_kpt[seq][i]
            pred_pose_3d = preds_3d_kpt[seq][i]

            pred_visibility = np.ones(pred_pose_2d.shape[:2], dtype='bool')

            # matching between 2D points
            if keep_matching:
                pair_inds = np.arange(gt['annot2'].shape[1])[gt['isValidFrame'][i]]
            else:
                pair_inds = _match_poses(gt_pose_2d[:, joints_for_matching], gt_visibility[:, joints_for_matching],
                                         pred_pose_2d[:, joints_for_matching], pred_visibility[:, joints_for_matching],
                                         40)

            my_matching_inds.append(pair_inds)
            has_pair = pair_inds >= 0

            # Reorder predicted poses to match Gt poses. If a GT pose does not have a pair, it is filled with 1e5
            reordered_pose_3d = 100000 * np.ones_like(gt_pose_3d)  # (nGtPoses, nJoints, 3)
            reordered_pose_3d[has_pair] = pred_pose_3d[pair_inds[has_pair]]  # (nGtPoses, nJoints, 3)

            gt_poses.append(gt_pose_3d)
            pred_poses.append(reordered_pose_3d)
            valid_pred.append(has_pair)

        gt_poses = np.concatenate(gt_poses)
        pred_poses = np.concatenate(pred_poses)
        valid_pred = np.concatenate(valid_pred)

        if is_relative:
            hip_ind = MuPoTSJoints().index_of('hip')
            gt_poses -= gt_poses[:, [hip_ind]]
            pred_poses -= pred_poses[:, [hip_ind]]

        # calculating per joint errors
        pred_poses = _scale_to_gt(pred_poses, gt_poses)
        pred_poses[~valid_pred] = 100000
        errors = np.linalg.norm(gt_poses - pred_poses, axis=2)  # (nGtPoses, nJoints)
        if matched_only:
            errors = errors[valid_pred]

        pck_by_sequence[seq] = np.mean(errors[:, scored_joints] < 150) * 100
        auc_by_sequence[seq] = np.mean([np.mean(errors[:, scored_joints] < t) for t in AUC_THRESHOLDS]) * 100
        all_perjoint_errors[seq] = errors

    return pck_by_sequence, auc_by_sequence
