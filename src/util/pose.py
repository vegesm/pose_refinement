import numpy as np

from databases.joint_sets import CocoExJoints
from util.misc import assert_shape


def harmonic_mean(a, b, eps=1e-6):
    return 2 / (1 / (a + eps) + 1 / (b + eps))


def _combine(data, target, a, b):
    """
    Modifies data by combining (taking average) joints at index a and b at position target.
    """
    data[:, target, :2] = (data[:, a, :2] + data[:, b, :2]) / 2
    data[:, target, 2] = harmonic_mean(data[:, a, 2], data[:, b, 2])


def extend_hrnet_raw(raw):
    """
    Adds the hip and neck to a Coco skeleton by averaging left/right hips and shoulders.
    The score will be the harmonic mean of the two.
    """
    assert_shape(raw, (None, 17, 3))
    js = CocoExJoints()

    result = np.zeros((len(raw), 19, 3), dtype='float32')
    result[:, :17, :] = raw
    _combine(result, js.index_of('hip'), js.index_of('left_hip'), js.index_of('right_hip'))
    _combine(result, js.index_of('neck'), js.index_of('left_shoulder'), js.index_of('right_shoulder'))

    return result


def insert_zero_joint(data, ind):
    """ Adds back a root with zeros in a hip-relative pose.

     :param ind: the root will be inserted here
     """
    assert data.ndim >= 2

    shape = list(data.shape)
    shape[-2] += 1
    result = np.zeros(shape, dtype=data.dtype)
    result[..., :ind, :] = data[..., :ind, :]
    result[..., ind + 1:, :] = data[..., ind:, :]

    return result


def remove_root(data, root_ind):
    """
    Removes a joint from a dataset by moving it to the origin and removing it from the array.

    :param data: (..., nJoints, 2|3) array
    :param root_ind: index of the joint to be removed
    :return: (..., nJoints-1, 2|3) array
    """
    assert data.ndim >= 2 and data.shape[-1] in (2, 3)

    roots = data[..., [root_ind], :]  # (..., 1, [2|3])
    data = data - roots
    data = np.delete(data, root_ind, axis=-2)

    return data


def remove_root_keepscore(data, root_ind):
    """
    Removes a joint from a 2D dataset by moving to the origin and removing it from the array.
    The difference to remove_root is that the third column stores the confidence score and it is
    not changed.

    :param data: (nPoses, nJoints, 3[x,y,score]) array
    :param root_ind: index of the joint to be removed
    :return: (nPoses, nJoints-1, 3[x,y,score]) array
    """
    assert data.ndim >= 3 and data.shape[-1] == 3, data.shape

    roots = data[..., [root_ind], :2]  # ndarray(...,1,2)
    # roots = roots.reshape((len(roots), 1, 2))
    data[..., :2] = data[..., :2] - roots
    data = np.delete(data, root_ind, axis=-2)

    return data


def combine_pose_and_trans(data3d, std3d, mean3d, joint_set, root_name, log_root_z=True):
    """
    3D result postprocess: unnormalizes data3d and reconstructs the absolute pose from relative + absolute split.

    Parameters:
        data3d: output of the PyTorch model, ndarray(nPoses, 3*nJoints), in the format created by preprocess3d
        std3d: normalization standard deviations
        mean3d: normalization means
        root_name: name of the root joint
        log_root_z: The z coordinate of the depth is in logarithms

    Returns:
        ndarray(nPoses, nJoints, 3)
    """
    assert_shape(data3d, (None, joint_set.NUM_JOINTS * 3))

    data3d = data3d * std3d + mean3d
    root = data3d[:, -3:]
    rel_pose = data3d[:, :-3].reshape((len(data3d), joint_set.NUM_JOINTS - 1, 3))

    if log_root_z:
        root[:, 2] = np.exp(root[:, 2])

    rel_pose += root[:, np.newaxis, :]

    result = np.zeros((len(data3d), joint_set.NUM_JOINTS, 3), dtype='float32')
    root_ind = joint_set.index_of(root_name)
    result[:, :root_ind, :] = rel_pose[:, :root_ind, :]
    result[:, root_ind, :] = root
    result[:, root_ind + 1:, :] = rel_pose[:, root_ind:, :]

    return result


def pose_interp(poses, good_frames):
    """
    Interpolates invisible poses.

    :param poses: (nPoses, nJoints, 3), the joint coordinates
    :param good_frames: (nPoses), true if the pose is detected on the frame, false otherwise
    :return: (nPoses, nHoints, 3), the inp
    """
    assert len(poses) == len(good_frames)
    assert poses.ndim == 3
    poses = poses.copy()

    frame_inds = np.arange(len(poses))
    for i in range(poses.shape[1]):
        for j in range(poses.shape[2]):
            # interpolate poses[:,i,j]
            poses[~good_frames, i, j] = np.interp(
                frame_inds[~good_frames], frame_inds[good_frames], poses[good_frames, i, j])

    return poses


HEIGHT_BONES = [['left_ankle', 'left_knee'], ['left_hip', 'left_knee'], ['hip', 'spine'], ['spine', 'neck']]


def _calc_limb_length(poses, joint_set, bones):
    """
    calculates the length of a limb that contains multiple bones.
    :param bones: list of (joint1, joint2) pairs, where joint1 and joint2 determines the bone.

    :return: For each pose, the sum of the lengths of the bones in `bones`
    """
    assert_shape(poses, ('*', joint_set.NUM_JOINTS, 3))
    bone_inds = [[joint_set.index_of(j) for j in b] for b in bones]

    height = np.zeros(poses.shape[:-2], dtype='float32')
    for bone in bone_inds:
        bones = poses[..., bone[0], :] - poses[..., bone[1], :]  # (shapePose, 3)
        bones = np.linalg.norm(bones, axis=-1)  # (shapePose)
        height += bones

    return height


def pck(pred, gt, thresh):
    """ Percentage of keypoints less than thresh mm away from the GT. """
    return np.mean(np.linalg.norm(pred - gt, axis=-1) < thresh)


AUC_THRESHOLDS = np.arange(0, 151, 5)


def auc(pred, gt, thresholds=AUC_THRESHOLDS):
    """ Calculates AUC of PCK. The default thresholds are the ones used by the MuPoTS evaluation script"""
    errors = np.linalg.norm(pred - gt, axis=-1)
    return np.mean([np.mean(errors < t) for t in AUC_THRESHOLDS])


def mpjpe(pred, gt):
    assert_shape(pred, ('*', None, 3))
    assert pred.shape == gt.shape

    return np.mean(np.linalg.norm(gt - pred, axis=-1))


def r_mpjpe(pred, gt, joint_set):
    pred = remove_root(pred, joint_set.index_of('hip'))
    gt = remove_root(gt, joint_set.index_of('hip'))

    return mpjpe(pred, gt)


def mrpe(pred, gt, joint_set):
    """ Mean Roo Position Error. """
    assert_shape(pred, ('*', None, 3))
    assert pred.shape == gt.shape
    hip_ind = joint_set.index_of('hip')

    assert gt[..., hip_ind, :].shape[-1] == 3
    return np.nanmean(np.linalg.norm(gt[..., hip_ind, :] - pred[..., hip_ind, :], axis=-1))


def optimal_scaling(pred, gt):
    """
    Calculates optimal scaling factor for a given set of points. Optimal scaling is the scalar s,
    with which the pred points scaled become the closest to gt points, in L2 sense.

    :param pred: array(nFrames, nPoints, 3)
    :param gt: array(nFrames, nPoints, 3)
    :return: array(nFrames,3)
    """
    assert pred.shape == gt.shape
    assert_shape(pred, ('*', None, 3))

    # Optimal scale transform
    dot_pose_pose = np.sum(pred * pred, axis=(-1, -2))  # (nShape) torch.sum(torch.mul(pred,pred),1,keepdim=True)
    dot_pose_gt = np.sum(pred * gt, axis=(-1, -2))

    return dot_pose_gt / dot_pose_pose  # (nShape), the optimal scaling factor s


def rn_mpjpe(pred, gt, root_ind):
    """
    N-MPJPE, when optimal scaling factor is calculated on relative pose.
    This hsould be a good comparison to height based scaling
    Based on https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning, losses/poses.py

    """
    assert pred.shape == gt.shape
    assert_shape(pred, ('*', None, 3))

    s_opt = optimal_scaling(remove_root(pred, root_ind), remove_root(gt, root_ind))

    return mpjpe(pred * s_opt[..., np.newaxis, np.newaxis], gt)


def n_mpjpe(pred, gt):
    """
    Based on https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning, losses/poses.py

    """
    assert pred.shape == gt.shape
    assert_shape(pred, ('*', None, 3))

    s_opt = optimal_scaling(pred, gt)

    return mpjpe(pred * s_opt[..., np.newaxis, np.newaxis], gt)
