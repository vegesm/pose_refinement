import numpy as np
import torch
from scipy import ndimage

from databases.joint_sets import MuPoTSJoints
from training.callbacks import BaseMPJPECalculator
from training.torch_tools import get_optimizer
from util.misc import assert_shape
from util.pose import remove_root, insert_zero_joint


def pose_error(pred, init):
    return torch.sum((pred - init) ** 2)


def euc_err(pred, gt):
    """ Calculates the euclidean distance between each joint (not squared). """
    if isinstance(pred, np.ndarray):
        return np.linalg.norm(pred - gt, axis=-1)
    else:
        return torch.norm(pred - gt, dim=-1)


def zero_velocity_loss(pred, step=1):
    return torch.sum((pred[step:] - pred[:-step]) ** 2)


def step_zero_velocity_loss(pred, step=1):
    return torch.sum((pred[step:] - pred[:-step]) ** 2, dim=(1, 2))


def const_velocity_loss(pred, step=1):
    velocity = pred[step:] - pred[:-step]
    return torch.sum((velocity[step:] - velocity[:-step]) ** 2)


def step_const_velocity_loss(pred, step):
    velocity = pred[step:] - pred[:-step]
    return torch.sum((velocity[step:] - velocity[:-step]) ** 2, dim=(1, 2))


def gmloss(err, a):
    """ Geman-McClure cost function"""
    square = err * err
    return square / (square + a)


def capped_l2(err, a):
    """ calculates min(err*2, a) """
    if isinstance(err, np.ndarray):
        return np.minimum(err * err, a)
    else:
        err2 = err * err
        #         err2 = err**2
        return torch.where(err2 < a, err2, a)


def capped_l2_euc_err(pred, gt, a):
    """ calculates min(err*2, a) """
    if isinstance(pred, np.ndarray):
        err = np.sum((pred - gt) ** 2, axis=-1)
        return np.minimum(err * err, a)
    else:
        diff = pred - gt
        err = torch.sum(diff * diff, dim=-1)
        return torch.where(err < a, err, a)


def abs_to_hiprel(poses, joint_set):
    """ Converts an absolute pose into [hi]+relative_pose. """
    assert_shape(poses, (None, joint_set.NUM_JOINTS, 3))

    root = poses[:, [joint_set.index_of('hip')]].copy()
    rel = remove_root(poses, joint_set.index_of('hip'))

    return np.concatenate([root, rel], axis=-2)


def add_back_hip(poses, joint_set):
    """ Inverse of abs_to_hiprel """
    assert_shape(poses, (None, joint_set.NUM_JOINTS, 3))
    root = poses[:, [0]].copy()

    hip_ind = joint_set.index_of('hip')
    result = insert_zero_joint(poses[:, 1:], hip_ind)
    result += root

    return result


class StackedArrayAllMupotsEvaluator(BaseMPJPECalculator):
    """
    An evaluator that expects a stacked numpy array as prediction results.
    Uses all poses, no masking out invisible poses.
    """

    def __init__(self, pred, dataset, ignore_invalid, post_process3d=None, prefix='test'):
        self.prediction = pred
        self.dataset = dataset
        self.ignore_invalid = ignore_invalid

        data_3d_mm = {}
        for seq in range(1, 21):
            inds = self.dataset.index.seq_num == seq
            if self.ignore_invalid:
                inds = inds & self.dataset.valid_annotations

            data_3d_mm[seq] = dataset.poses3d[inds]

        super().__init__(data_3d_mm, dataset.pose3d_jointset, post_process3d=post_process3d, csv=None, prefix=prefix)

    def pred_and_calc_loss(self, model):
        assert model is None, "StackedArrayAllMupotsEvaluator does not handle model evaluation"
        preds = {}
        losses = {}
        for seq in range(1, 21):
            inds = self.dataset.index.seq_num == seq
            if self.ignore_invalid:
                inds = inds & self.dataset.valid_annotations

            preds[seq] = self.prediction[inds]
            losses[seq] = np.zeros_like(preds[seq])

        return losses, preds


def optimize_poses(pred3d, data, _config, **kwargs):
    """
    Runs the optimisation process on the dataset defined by resulsts.
    Parameters:
        pred3d: poses predicted by VideoPose, aligned with dataset
        dataset: dataset describing
        _config: dictionary of additional parameters
    """
    _config = dict(_config)
    _config.update(kwargs)

    joint_set = MuPoTSJoints()

    seqs = np.unique(data.index.seq)
    smoothed_pred = np.zeros_like(pred3d)

    losses = []

    for seq in seqs:
        inds = data.index.seq == seq

        poses_init = abs_to_hiprel(pred3d[inds].copy(), joint_set).astype('float32') / 1000

        # interpolate invisible poses, if required
        poses_pred = poses_init.copy()

        kp_score = np.mean(data.poses2d[inds, :, 2], axis=-1)
        if _config['smooth_visibility']:
            kp_score = ndimage.median_filter(kp_score, 9)
        kp_score = torch.from_numpy(kp_score).cuda()
        poses_init = torch.from_numpy(poses_init).cuda()
        poses_pred = torch.from_numpy(poses_pred).cuda()
        scale = torch.ones((len(kp_score), 1, 1))

        poses_init.requires_grad = False
        poses_pred.requires_grad = True
        kp_score.requires_grad = False
        scale.requires_grad = False

        optimizer = get_optimizer([poses_pred], _config)

        for i in range(_config['num_iter']):
            # smoothing formulation
            if _config['pose_loss'] == 'gm':
                pose_loss = torch.sum(kp_score.view(-1, 1, 1) * gmloss(poses_pred - poses_init, _config['gm_alpha']))
            elif _config['pose_loss'] == 'capped_l2':
                pose_loss = torch.sum(kp_score.view(-1, 1, 1) * capped_l2(poses_pred - poses_init,
                                                                          torch.tensor(_config['l2_cap']).float().cuda()))
            elif _config['pose_loss'] == 'capped_l2_euc_err':
                pose_loss = torch.sum(kp_score.view(-1, 1) * capped_l2_euc_err(poses_pred, poses_init,
                                                                               torch.tensor(_config['l2_cap']).float().cuda()))
            else:
                raise NotImplementedError('Unknown pose_loss' + _config['pose_loss'])

            velocity_loss_hip = torch.sum(globals()[_config['smoothness_loss_hip']](poses_pred[:, [0], :], 1))

            step = _config['smoothness_loss_hip_largestep']
            vel_loss = globals()[_config['smoothness_loss_hip']](poses_pred[:, [0], :], step)
            velocity_loss_hip_large = torch.sum((1 - kp_score[-len(vel_loss):]) * vel_loss)

            velocity_loss_rel = torch.sum(globals()[_config['smoothness_loss_rel']](poses_pred[:, 1:, :], 1))
            vel_loss = globals()[_config['smoothness_loss_rel']](poses_pred[:, 1:, :], step)
            velocity_loss_rel_large = torch.sum((1 - kp_score[-len(vel_loss):]) * vel_loss)

            total_loss = pose_loss + _config['smoothness_weight_hip'] * velocity_loss_hip \
                         + _config['smoothness_weight_hip_large'] * velocity_loss_hip_large \
                         + _config['smoothness_weight_rel'] * velocity_loss_rel \
                         + _config['smoothness_weight_rel_large'] * velocity_loss_rel_large

            optimizer.zero_grad()
            total_loss.backward()

            optimizer.step()

        poses_init = poses_init.detach().cpu().numpy() * 1000
        poses_pred = poses_pred.detach().cpu().numpy() * 1000

        poses_init = add_back_hip(poses_init, joint_set)
        poses_pred = add_back_hip(poses_pred, joint_set)
        smoothed_pred[inds] = poses_pred

        losses.append(total_loss.item())

    if _config.get('print_loss', False):
        print('Avg loss:', np.mean(losses))
    return smoothed_pred
