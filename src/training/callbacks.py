import math
import numpy as np
import torch

from training.loaders import UnchunkedGenerator
from training.torch_tools import eval_results
from util.pose import remove_root, mrpe, optimal_scaling, r_mpjpe


class BaseCallback(object):

    def on_itergroup_end(self, iter_cnt, epoch_loss):
        pass

    def on_epoch_end(self, model, epoch, epoch_loss, optimizer, epoch_vals):
        pass


def _sample_value(dictionary):
    """ Selects a value from a dictionary, it is always the same element. """
    return list(dictionary.values())[0]


class BaseMPJPECalculator(BaseCallback):
    """
    Base class for calculating and displaying MPJPE stats, grouped by something (sequence most of the time).
    """
    PCK_THRESHOLD = 150

    def __init__(self, data_3d_mm, joint_set, post_process3d=None, csv=None, prefix='val'):
        """

        :param data_3d_mm: dict, group_name-> ndarray(n.Poses, nJoints, 3). The ground truth poses in mm.
        """
        self.csv = csv
        self.prefix = prefix
        self.pctiles = [5, 10, 50, 90, 95, 99]

        if self.csv is not None:
            with open(csv, 'w') as f:
                f.write('epoch,type,name,avg')
                f.write(''.join([',pct' + str(x) for x in self.pctiles]))
                f.write('\n')

        self.data_3d_mm = data_3d_mm

        self.is_absolute = _sample_value(self.data_3d_mm).shape[1] == joint_set.NUM_JOINTS
        self.num_joints = joint_set.NUM_JOINTS if self.is_absolute else joint_set.NUM_JOINTS - 1

        self.joint_set = joint_set
        self.post_process3d = post_process3d
        self.sequences = sorted(list(data_3d_mm.keys()))

    def on_epoch_end(self, model, epoch, epoch_loss, optimizer, epoch_vals):
        sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles = self.eval(model, verbose=True)

        if self.csv is not None:
            joint_names = self.joint_set.NAMES.copy()
            if not self.is_absolute:
                joint_names = np.delete(joint_names, self.joint_set.index_of('hip'))  # remove root

            with open(self.csv, 'a') as f:
                for seq in self.sequences:
                    f.write('%d,%s,%s,%f' % (epoch, 'sequence', seq, sequence_mpjpes[seq]))
                    for i in range(len(self.pctiles)):
                        f.write(',%f' % sequence_pctiles[seq][i])
                    f.write('\n')

                for joint_id in range(self.num_joints):
                    f.write('%d,%s,%s,%f' % (epoch, 'joint', joint_names[joint_id], joint_means[joint_id]))
                    for i in range(len(self.pctiles)):
                        f.write(',%f' % joint_pctiles[i, joint_id])
                    f.write('\n')

    def eval(self, model=None, calculate_scale_free=False, verbose=False):
        """
        :param model: the evaluator can use this model, if self.model is nor provided
        :param calculate_scale_free: if True, also calculates N-MRPE and N_RMPJPE
        :return:
        """
        losses, preds = self.pred_and_calc_loss(model)
        losses = np.concatenate([losses[seq] for seq in self.sequences])
        self.val_loss = np.nanmean(losses)
        self.losses_to_log = {self.prefix + '_loss': self.val_loss}

        self.losses = losses
        self.preds = preds

        # Assuming hip is the last component
        if self.is_absolute:
            self.losses_to_log[self.prefix + '_abs_loss'] = np.nanmean(losses[:, -3:])
            self.losses_to_log[self.prefix + '_rel_loss'] = np.nanmean(losses[:, :-3])
        else:
            self.losses_to_log[self.prefix + '_rel_loss'] = self.val_loss

        assert self.pctiles[-1] == 99, "Currently the last percentile is hardcoded to be 99 for printing"

        sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles = \
            eval_results(preds, self.data_3d_mm, self.joint_set, pctiles=self.pctiles, verbose=verbose)
        self.losses_to_log[self.prefix + '_mrpe'] = np.mean([mrpe(preds[s], self.data_3d_mm[s], self.joint_set)
                                                             for s in preds])

        # Calculate relative error
        if self.is_absolute:
            rel_pred = {}
            rel_gt = {}
            for seq in preds:
                rel_pred[seq] = remove_root(preds[seq], self.joint_set.index_of('hip'))
                rel_gt[seq] = remove_root(self.data_3d_mm[seq], self.joint_set.index_of('hip'))
            rel_mean_error, _, _, _, _ = eval_results(rel_pred, rel_gt, self.joint_set, verbose=False)
            rel_mean_error = np.mean(np.asarray(list(rel_mean_error.values()), dtype=np.float32))
            if verbose:
                print("Root relative error (MPJPE): %.2f" % rel_mean_error)
            self.rel_mean_error = rel_mean_error
            self.losses_to_log[self.prefix + '_rel_error'] = rel_mean_error

        self.mean_sequence_mpjpe = np.mean(np.asarray(list(sequence_mpjpes.values()), dtype=np.float32))
        self.mean_sequence_pck = np.mean(np.asarray(list(sequence_pcks.values()), dtype=np.float32))
        self.losses_to_log[self.prefix + '_err'] = self.mean_sequence_mpjpe
        self.losses_to_log[self.prefix + '_pck'] = self.mean_sequence_pck

        if calculate_scale_free:
            scaled_preds = {}
            for seq in preds:
                # predict a single scale for the full video
                pred_points = preds[seq].reshape(1, -1, 3)
                gt_points = self.data_3d_mm[seq].reshape(1, -1, 3)

                s = optimal_scaling(pred_points, gt_points)
                scaled_preds[seq] = preds[seq] * s

            n_mrpe = np.mean([mrpe(scaled_preds[s], self.data_3d_mm[s], self.joint_set) for s in scaled_preds])
            n_rmpjpe = np.mean([r_mpjpe(scaled_preds[s], self.data_3d_mm[s], self.joint_set) for s in scaled_preds])

            if verbose:
                print('N-MRPE: %.1f' % n_mrpe)
                print('N-MPJPE: %.1f' % n_rmpjpe)
            self.losses_to_log[self.prefix + '_n_mrpe'] = n_mrpe
            self.losses_to_log[self.prefix + '_n_rel_err'] = n_rmpjpe

        return sequence_mpjpes, sequence_pcks, sequence_pctiles, joint_means, joint_pctiles

    def results_and_gt(self):
        """
        Returns the gt and result matrices as list of (seq, pred, gt) tuples
        """
        keys = sorted(list(self.data_3d_mm.keys()))
        return [(seq, self.preds[seq], self.data_3d_mm[seq]) for seq in keys]

    def pred_and_calc_loss(self, model):
        """
        Subclasses must implement this method. It calculates the loss
        and the predictions of the current model.

        :param model: model received in the on_epoch_end callback
        :return: (loss, pred) pair, each is a dictionary from sequence name to loss or prediction
        """
        raise NotImplementedError()


class TemporalTestEvaluator(BaseMPJPECalculator):
    """ Can be used with MPII-3DHP dataset to create"""

    def __init__(self, model, dataset, loss, augment, post_process3d=None, prefix='test'):
        self.model = model
        self.dataset = dataset
        self.augment = augment
        pad = (model.receptive_field() - 1) // 2
        self.generator = UnchunkedGenerator(dataset, pad, self.augment)
        self.seqs = sorted(np.unique(dataset.index.seq))

        data_3d_mm = {}
        self.preprocessed3d = {}
        for seq in self.seqs:
            inds = np.where(dataset.index.seq == seq)[0]
            batch = dataset.get_samples(inds, False)
            self.preprocessed3d[seq] = batch['pose3d'][batch['valid_pose']]
            data_3d_mm[seq] = dataset.poses3d[inds][batch['valid_pose']]

        if loss == 'l1' or loss == 'l1_nan':
            self.loss = lambda p, t: np.abs(p - t)
        elif loss == 'l2':
            self.loss = lambda p, t: np.square(p - t)

        super().__init__(data_3d_mm, dataset.pose3d_jointset, post_process3d=post_process3d, csv=None, prefix=prefix)

    def pred_and_calc_loss(self, model):
        """
        Subclasses must implement this method. It calcula
        :param model: model received in the on_epoch_end callback
        :return: (loss, pred) pair, each is a dictionary from sequence name to loss or prediction
        """
        preds = {}
        self.raw_preds = {}
        losses = {}
        with torch.no_grad():
            for i, (pose2d, valid) in enumerate(self.generator):
                seq = self.seqs[i]
                pred3d = self.model(torch.from_numpy(pose2d).cuda()).detach().cpu().numpy()
                self.raw_preds[seq] = pred3d.copy()

                valid = valid[0]
                losses[seq] = self.loss(pred3d[0][valid], self.preprocessed3d[seq])

                pred_real_pose = self.post_process3d(pred3d[0], seq)  # unnormalized output

                if self.augment:
                    pred_real_pose_aug = self.post_process3d(pred3d[1], seq)
                    pred_real_pose_aug[:, :, 0] *= -1
                    pred_real_pose_aug = self.dataset.pose3d_jointset.flip(pred_real_pose_aug)
                    pred_real_pose = (pred_real_pose + pred_real_pose_aug) / 2

                preds[seq] = pred_real_pose[valid]

        return losses, preds


class TemporalMupotsEvaluator(TemporalTestEvaluator):
    """ Can be used with PersonStackedMupots dataset for a temporal model.  """

    def __init__(self, model, dataset, loss, augment, post_process3d=None, prefix='test'):
        super().__init__(model, dataset, loss, augment, post_process3d=post_process3d, prefix=prefix)

        self.data_3d_mm = TemporalMupotsEvaluator._group_by_seq(self.data_3d_mm)
        self.sequences = sorted(self.data_3d_mm.keys())

    @staticmethod
    def _group_by_seq(data):
        per_person_keys = sorted(data.keys())
        result = {}
        for seq in range(1, 21):
            keys = sorted([k for k in per_person_keys if k.startswith('%d/' % seq)])
            assert len(keys) > 0, per_person_keys
            result[seq] = np.concatenate([data[k] for k in keys])

        return result

    def pred_and_calc_loss(self, model):
        losses, preds = super().pred_and_calc_loss(model)
        losses = TemporalMupotsEvaluator._group_by_seq(losses)
        preds = TemporalMupotsEvaluator._group_by_seq(preds)

        return losses, preds


class ModelCopyTemporalEvaluator(TemporalTestEvaluator):
    """
    Same as TemporalTestEvaluator but uses another model for evaluation than for training,
    and before evaluation copies the weights to the 'eval' model
    """

    def pred_and_calc_loss(self, train_model):
        """ train_model is coming from the training loop """
        self.model.load_state_dict(train_model.state_dict())
        self.model.eval()

        return super().pred_and_calc_loss(None)


def preds_from_logger(dataset, logger):
    """
    Arranges results from LogAllMillimeterCallback according to index in dataset
    """
    # Special handling for multipose inputs
    if dataset.poses3d.ndim == 4:
        pose_shape = list(logger.data_3d_mm.values())[0].shape
        result = np.zeros((dataset.poses3d.shape[:2]) + pose_shape[1:])
        result[:] = np.nan

        seqs = np.unique(dataset.index.seq)
        for seq in seqs:
            inds = dataset.index.seq == seq
            mask = np.zeros(result.shape[:2], dtype='bool')
            assert np.all(~mask)
            mask[inds] = dataset.good_poses[inds]  # composing masks
            result[mask] = logger.preds[seq]

        return result

    elif dataset.poses3d.ndim == 3:
        pose_shape = list(logger.data_3d_mm.values())[0].shape
        result = np.zeros((len(dataset.index),) + pose_shape[1:])

        seqs = np.unique(dataset.index.seq)
        for seq in seqs:
            inds = dataset.index.seq == seq
            mask = np.zeros(len(result), dtype='bool')
            mask[inds] = dataset.good_poses[inds]  # composing masks
            result[mask] = logger.preds[seq]

        return result
    else:
        raise Exception("unexpected shape")


class ModelSaver(BaseCallback):
    """
    Saves the best model at every epoch.
    %d in the path can specify the epoch
    """

    def __init__(self, path):
        self.path = path

    def on_epoch_end(self, model, epoch, epoch_loss, optimizer, epoch_vals):
        path = self.path
        if '%d' in path:
            path = path % epoch

        torch.save(model.state_dict(), path)


class BestModelSaver(BaseCallback):
    """
    Saves the best model according to a given metric.
    Useful together with early stopping.
    """

    def __init__(self, path, evaluator, metric, lower_better=True):
        assert lower_better, "lower_better=False not implemented yet"
        self.path = path
        self.evaluator = evaluator
        self.metric = metric
        self.best_value = math.inf

    def on_epoch_end(self, model, epoch, epoch_loss, optimizer, epoch_vals):
        path = self.path
        if '%d' in path:
            path = path % epoch

        if self.evaluator.losses_to_log[self.metric] < self.best_value:
            self.best_value = self.evaluator.losses_to_log[self.metric]
            torch.save(model.state_dict(), path)
