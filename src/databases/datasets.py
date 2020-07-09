import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from databases import mupots_3d, mpii_3dhp, muco_temp
from databases.joint_sets import CocoExJoints, OpenPoseJoints, MuPoTSJoints


class PoseDataset(Dataset):
    """ Subclasses should have the attributes poses2d/3d, pred_cdepths, pose[2|3]d_jointset defined."""

    def filter_dataset(self, inds):
        """
        Filters the dataset by ``inds``.

        :param inds: anything that can be used for numpy masking
        """
        if hasattr(self, 'coord_depths'):
            self.coord_depths = self.coord_depths[inds]
        if hasattr(self, 'width'):
            self.width = self.width[inds]

        self.pred_cdepths = self.pred_cdepths[inds]
        self.poses2d = self.poses2d[inds]
        self.poses3d = self.poses3d[inds]

        self.gt_depth_2d = None  # self.gt_depth_2d[good_poses], gt_depth_2d is not implemented for compressed videos,
        # and is is ignored for CoordDepthDataset anywat
        self.index = self.index[inds]
        self.fx = self.fx[inds]
        self.fy = self.fy[inds]
        self.cx = self.cx[inds]
        self.cy = self.cy[inds]


class AugmentMixin:
    def augment(self, scale_by_dist, scales=None):
        """
        Augments the data in a pose dataset. It simulates moving the poses
        closer and further away from the camera. The method takes the dataset D, applies a transformation T,
        and concatenates the transformed data to the original data.

        :param scale_by_dist: If true, during augmentation it scales values with l2 distance from camera,
                              otherwise with z coordinate (depth).
        :param scales: if defined, values in this array used for scaling instead of random values
        """
        assert isinstance(self.pose3d_jointset, MuPoTSJoints), "only implemented for MupoTS joints"
        orig_size = len(self.poses2d)
        root_ind = MuPoTSJoints().index_of('hip')

        # Calculating minimum scale to avoid joints behind camera
        if scales is None:
            limb_vec = self.poses3d[:, :, 2] - self.poses3d[:, [root_ind], 2]
            min_scale = np.nanmax(-limb_vec / self.poses3d[:, [root_ind], 2], axis=1)

            scales = np.random.normal(1, 0.25, orig_size)
            scales[scales < 0.6] = 1
            scales = np.maximum(scales, min_scale + 1e-5)
            scales[scales > 1.5] = 1
            scales = scales.reshape((-1, 1))
        else:
            assert scales.ndim == 2, "scales is expected to be a column vector"
        self.scales = scales.copy()

        # Duplicate all the training data, the first half is the original unchanged,
        # the second half is augmented
        for field in ['poses2d', 'poses3d', 'pred_cdepths', 'fx', 'fy', 'cx', 'cy', 'width', 'valid_2d_pred']:
            if hasattr(self, field):
                data = self.__getattribute__(field)
                self.__setattr__(field, np.concatenate([data, data.copy()]))
        if hasattr(self, 'index'):
            self.index = np.concatenate([self.index, self.index.copy()])

        # Calculate the new 3D coordinates of the poses
        orig_roots = np.expand_dims(self.poses3d[orig_size:, root_ind, :].copy(), 1)  # (nPoses, 1, 3)
        new_roots = orig_roots * np.expand_dims(scales, 1)
        self.poses3d[orig_size:, :, :] = self.poses3d[orig_size:, :, :] - orig_roots + new_roots

        pose2d_root_ind = self.pose2d_jointset.index_of('hip')
        self.poses2d[orig_size:, :, :2] = (self.poses2d[orig_size:, :, :2]
                                           - self.poses2d[orig_size:, [pose2d_root_ind], :2]) / scales[:, :, None] \
                                          + self.poses2d[orig_size:, [pose2d_root_ind], :2]

        assert np.all((self.poses3d[:, :, 2] >= 0) | np.isnan(self.poses3d[:, :, 2])), "Joint behind camera"


class TemporalAugmentMixin(AugmentMixin):

    def augment(self, scale_by_dist, scales=None):
        orig_len = len(self.poses2d)
        if scales is None:
            # creating scales such that poses on a single frame have the same scale
            root_ind = self.pose3d_jointset.index_of('hip')
            limb_vec = self.poses3d[:, :, 2] - self.poses3d[:, [root_ind], 2]
            min_scales = np.nanmax(-limb_vec / self.poses3d[:, [root_ind], 2], axis=1)

            scales = np.ones(len(self.poses2d), dtype='float32')
            seqs = sorted(np.unique(self.index.seq))
            for seq in seqs:
                inds = self.index.seq == seq
                # print(np.sum(inds), seq, self.index.seq)
                min_scale = np.max(min_scales[inds])

                scale = np.random.normal(1, 0.2)
                scale = max(scale, 0.6)
                scale = max(scale, min_scale + 1e-5)
                scale = min(scale, 1.5)

                scales[inds] = scale

            scales = scales[:, np.newaxis]

        super().augment(scale_by_dist, scales)
        self.index = np.rec.array(self.index)
        for i in range(orig_len, 2 * orig_len):
            self.index.seq[i] = self.index.seq[i] + 'A'


class FlippableDataset(PoseDataset):
    def __len__(self):
        return len(self.poses2d)

    def get_samples(self, ind, flip):
        """
        :param ind: indices of the elements to extract
        :param flip: true if elements should be flipped all of them
        """
        sample = self.prepare_sample(ind)
        if isinstance(flip, np.ndarray) or flip:
            if not isinstance(flip, np.ndarray):
                flip = np.full(len(ind), flip, dtype='bool')

            pose2d = sample['pose2d'].copy()
            pose2d[flip, ..., 0] = np.expand_dims(sample['width'][flip], 1) - pose2d[flip, ..., 0]
            pose2d[flip] = self.pose2d_jointset.flip(pose2d[flip])
            sample['pose2d'] = pose2d

            pose3d = sample['pose3d'].copy()
            pose3d[flip, ..., 0] *= -1
            pose3d[flip] = self.pose3d_jointset.flip(pose3d[flip])
            sample['pose3d'] = pose3d

            cx = sample['cx'].copy()
            cx[flip] = sample['width'][flip] - cx[flip]
            sample['cx'] = cx

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, ind):
        return self.get_samples(ind, False)


class ConcatPoseDataset(FlippableDataset, TemporalAugmentMixin):

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

        fields = ['poses2d', 'poses3d', 'pred_cdepths', 'fx', 'fy', 'cx', 'cy', 'valid_2d_pred']
        for field in fields:
            field1 = data1.__getattribute__(field)
            field2 = data2.__getattribute__(field)
            self.__setattr__(field, np.concatenate([field1, field2]))
        seqs = np.concatenate([data1.index.seq, data2.index.seq])
        self.index = np.recarray(len(seqs), [('seq', seqs.dtype)])
        self.index.seq = seqs

        assert type(data1.pose2d_jointset) == type(data2.pose2d_jointset)
        assert type(data1.pose3d_jointset) == type(data2.pose3d_jointset)
        self.pose2d_jointset = data1.pose2d_jointset
        self.pose3d_jointset = data1.pose3d_jointset

        self.transform = None

    def prepare_sample(self, ind):
        if isinstance(ind, (list, tuple, np.ndarray)):
            width = np.full(len(ind), 2048, dtype='int32')
        else:
            width = 2048

        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind], 'pred_cdepth': self.pred_cdepths[ind],
                  'index': ind, 'valid_pose': self.valid_2d_pred[ind], 'cx': self.cx[ind], 'width': width}

        return sample


def _column_stack(data):
    """ Columnwise stacks an ndarray"""
    return data.reshape((-1,) + data.shape[2:], order='F').copy()


class PersonStackedMuPoTsDataset(FlippableDataset):
    def __init__(self, pose2d_type, pose3d_scaling, pose_validity='detected_only', hip_threshold=-1):
        """
        Loads MuPoTS dataset but only those images where at least one person was detected. Each person on a frame
        is loaded separately.

        :param pose_validity: one of 'all', 'detected_only', 'valid_only'; specifies which poses are marked valid
                              all - all of them; valid_only - those that are valid according to the GT annotations
                              detected_only - those that were successfuly detected by the 2D algon and also valid
        :param hip_threshold: only those poses are loaded, where the score of the hip is larger than this value
        :param filter_incorrect_match: MuPoTS's pose matching script has some erroneous matching. If filter_incorrect_match is True,
                                these are not loaded.
        """
        assert pose_validity in ['all', 'detected_only', 'valid_only']
        assert pose3d_scaling in ['univ', 'normal']

        self.pose2d_jointset = PersonStackedMuPoTsDataset.get_jointset(pose2d_type)
        self.pose3d_jointset = MuPoTSJoints()
        self.pose3d_scaling = pose3d_scaling
        pred2d_root_ind = self.pose2d_jointset.index_of('hip')

        poses2d = []
        poses3d = []
        joint3d_visible = []
        all_good_poses = []
        valid_annotations = []
        width = []
        index = []
        for seq in range(1, 21):
            img_width, img_height = mupots_3d.image_size(seq)

            gt = mupots_3d.load_gt_annotations(seq)
            pred2d = mupots_3d.load_2d_predictions(seq, pose2d_type)

            pose2d = pred2d['pose']
            pose3d = gt['annot3' if pose3d_scaling == 'normal' else 'univ_annot3']
            visibility = ~gt['occlusions']

            if pose_validity == 'all':
                good_poses = np.full(pose3d.shape[:2], True, dtype='bool')
            elif pose_validity == 'valid_only':
                good_poses = gt['isValidFrame'].squeeze()
            elif pose_validity == 'detected_only':
                good_poses = gt['isValidFrame'].squeeze()
                good_poses = np.logical_and(good_poses, pred2d['valid_pose'])
                good_poses = np.logical_and(good_poses, pose2d[:, :, pred2d_root_ind, 2] > hip_threshold)
            else:
                raise NotImplementedError("Unknown pose_validity value:" + pose_validity)

            orig_frame = np.tile(np.arange(len(good_poses)).reshape(-1, 1), (1, good_poses.shape[1]))
            orig_pose = np.tile(np.arange(good_poses.shape[1]).reshape(1, -1), (good_poses.shape[0], 1))

            assert pose2d.shape[:2] == good_poses.shape  # (nFrames, nPeople)
            assert pose3d.shape[:2] == good_poses.shape
            assert orig_frame.shape == good_poses.shape
            assert orig_pose.shape == good_poses.shape
            assert pose2d.shape[2:] == (self.pose2d_jointset.NUM_JOINTS, 3)
            assert pose3d.shape[2:] == (17, 3)
            assert visibility.shape[2] == 17
            assert good_poses.ndim == 2

            orig_frame = _column_stack(orig_frame)
            orig_pose = _column_stack(orig_pose)

            index.extend([('%d/%d' % (seq, orig_pose[i]), seq, orig_frame[i], orig_pose[i]) for i in range(len(orig_frame))])

            poses2d.append(_column_stack(pose2d))
            poses3d.append(_column_stack(pose3d))
            joint3d_visible.append(_column_stack(visibility))
            all_good_poses.append(_column_stack(good_poses))
            valid_annotations.append(_column_stack(gt['isValidFrame']))
            width.extend([img_width] * len(orig_frame))

        self.poses2d = np.concatenate(poses2d).astype('float32')
        self.poses3d = np.concatenate(poses3d).astype('float32')
        self.joint3d_visible = np.concatenate(joint3d_visible)
        self.good_poses = np.concatenate(all_good_poses)
        self.valid_annotations = np.concatenate(valid_annotations)
        self.width = np.array(width)
        self.index = np.rec.array(index, dtype=[('seq', 'U5'), ('seq_num', 'int32'), ('frame', 'int32'), ('pose', 'int32')])

        assert self.valid_annotations.shape == self.good_poses.shape
        assert len(self.valid_annotations) == len(self.poses2d)

        # Load calibration matrices
        N = len(self.poses2d)
        self.fx = np.zeros(N, dtype='float32')
        self.fy = np.zeros(N, dtype='float32')
        self.cx = np.zeros(N, dtype='float32')
        self.cy = np.zeros(N, dtype='float32')

        mupots_calibs = mupots_3d.get_calibration_matrices()
        for seq in range(1, 21):
            inds = (self.index.seq_num == seq)
            self.fx[inds] = mupots_calibs[seq][0, 0]
            self.fy[inds] = mupots_calibs[seq][1, 1]
            self.cx[inds] = mupots_calibs[seq][0, 2]
            self.cy[inds] = mupots_calibs[seq][1, 2]

        assert np.all(self.fx > 0), "Some fields were not filled"
        assert np.all(self.fy > 0), "Some fields were not filled"
        assert np.all(np.abs(self.cx) > 0), "Some fields were not filled"
        assert np.all(np.abs(self.cy) > 0), "Some fields were not filled"
        self.transform = None

    @staticmethod
    def get_jointset(pose2d_type):
        if pose2d_type == 'openpose':
            return OpenPoseJoints()
        elif pose2d_type == 'hrnet':
            return CocoExJoints()
        else:
            raise Exception("Unknown 2D pose type: " + pose2d_type)

    def prepare_sample(self, ind):
        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind],
                  'index': ind, 'valid_pose': self.good_poses[ind], 'cx': self.cx[ind], 'width': self.width[ind]}

        return sample


class Mpi3dTestDataset(FlippableDataset):
    def __init__(self, pose2d_type, pose3d_scaling, eval_frames_only=False):
        assert pose2d_type == 'hrnet', "Only hrnet 2d is implemented"
        assert pose3d_scaling in ['normal', 'univ'], \
            "Unexpected pose3d scaling type: " + str(pose3d_scaling)
        self.transform = None
        self.eval_frames_only = eval_frames_only

        pose3d_key = 'annot3' if pose3d_scaling == 'normal' else 'univ_annot3'

        poses2d = []
        poses3d = []
        valid_2d_pred = []  # True if HR-net found a pose
        valid_frame = []  # True if MPI-INF-3DHP marked the frame as valid
        fx = []
        fy = []
        cx = []
        cy = []
        width = []
        index = []

        for seq in range(1, 7):
            gt = h5py.File(os.path.join(mpii_3dhp.MPII_3DHP_PATH,
                                        'mpi_inf_3dhp_test_set', 'TS%d' % seq, 'annot_data.mat'), 'r')
            poses3d.append(gt[pose3d_key][:, 0])
            valid_frame.append(gt['valid_frame'][()] == 1)
            num_frames = len(poses3d[-1])  # The annotations are shorter than the number of images

            tmp = mpii_3dhp.test_poses_hrnet(seq)
            poses2d.append(tmp['poses'])
            valid_2d_pred.append(tmp['is_valid'])

            assert len(poses3d[-1]) == len(poses2d[-1]), "Gt and predicted frames are not aligned, seq:" + str(seq)

            index.extend([(seq, i) for i in range(num_frames)])

            calibration_mx = mpii_3dhp.get_test_calib(seq)
            fx.extend([calibration_mx[0, 0]] * num_frames)
            fy.extend([calibration_mx[1, 1]] * num_frames)
            cx.extend([calibration_mx[0, 2]] * num_frames)
            cy.extend([calibration_mx[1, 2]] * num_frames)
            width.extend([2048 if seq < 5 else 1920] * num_frames)

        self.pose2d_jointset = CocoExJoints()
        self.pose3d_jointset = MuPoTSJoints()

        self.poses2d = np.concatenate(poses2d)
        self.poses3d = np.concatenate(poses3d)
        self.valid_2d_pred = np.concatenate(valid_2d_pred)
        valid_frame = np.concatenate(valid_frame)
        assert valid_frame.shape[1] == 1, valid_frame.shape
        valid_frame = valid_frame[:, 0]
        self.index = np.rec.array(index, dtype=[('seq', 'int32'), ('frame', 'int32')])

        self.fx = np.array(fx, dtype='float32')
        self.fy = np.array(fy, dtype='float32')
        self.cx = np.array(cx, dtype='float32')
        self.cy = np.array(cy, dtype='float32')
        self.width = np.array(width, dtype='int32')

        assert len(self.poses2d) == len(self.index), len(self.index)

        # keep only those frame where a pose was detected
        good_poses = self.valid_2d_pred.copy()
        if eval_frames_only:
            good_poses = good_poses & valid_frame

        self.good_poses = good_poses

        assert len(self.poses2d) == len(self.poses3d)
        assert len(self.poses2d) == len(self.index), len(self.index)
        assert len(self.poses2d) == len(self.valid_2d_pred), len(self.valid_2d_pred)
        assert len(self.poses2d) == len(self.fx), len(self.fx)
        assert len(self.poses2d) == len(self.fy), len(self.fy)
        assert len(self.poses2d) == len(self.cx), len(self.cx)
        assert len(self.poses2d) == len(self.cy), len(self.cy)
        assert len(self.poses2d) == len(self.width), len(self.width)
        assert len(self.poses2d) == len(self.good_poses), len(self.good_poses)

    def prepare_sample(self, ind):
        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind],
                  'index': ind, 'valid_pose': self.good_poses[ind], 'cx': self.cx[ind], 'width': self.width[ind]}

        return sample


class Mpi3dTrainDataset(FlippableDataset, TemporalAugmentMixin):
    def __init__(self, pose2d_type, pose3d_scaling, cap_at_25fps, stride=1):
        assert pose2d_type == 'hrnet', "Only hrnet 2d is implemented"
        assert pose3d_scaling in ['normal', 'univ'], \
            "Unexpected pose3d scaling type: " + str(pose3d_scaling)
        self.transform = None

        pose3d_key = 'annot3' if pose3d_scaling == 'normal' else 'univ_annot3'

        poses2d = []
        poses3d = []
        valid_2d_pred = []  # True if HR-net found a pose
        fx = []
        fy = []
        cx = []
        cy = []
        index = []
        sequences = []

        calibs = mpii_3dhp.get_calibration_matrices()
        for sub in range(1, 9):
            for seq in range(1, 3):
                gt = mpii_3dhp.train_ground_truth(sub, seq)
                for cam in range(11):
                    # In S3/Seq2 cam2 there are some frame between 9400-9900 where the pose is
                    # behind the camera/nearly in the camera plane. This breaks training.
                    # For simplicity, ignore the whole set but ignoring frames 9400-9900
                    # would also work
                    if seq == 2 and sub == 3 and cam == 2:
                        continue

                    # Find indices that are selected for the dataset
                    inds = np.arange(len(gt[pose3d_key][cam]))
                    if cap_at_25fps and mpii_3dhp.get_train_fps(sub, seq) == 50:
                        inds = inds[::2]
                    inds = inds[::stride]
                    num_frames = len(inds)

                    poses3d.append(gt[pose3d_key][cam][inds])

                    tmp = mpii_3dhp.train_poses_hrnet(sub, seq, cam)
                    poses2d.append(tmp['poses'][inds])
                    valid_2d_pred.append(tmp['is_valid'][inds])

                    assert len(poses3d[-1]) == len(poses2d[-1]
                                                   ), "Gt and predicted frames are not aligned, seq:" + str(seq)

                    seq_name = 'S%d/Seq%d/%d' % (sub, seq, cam)
                    sequences.append(seq_name)
                    index.extend([(seq_name, sub, seq, cam, i) for i in inds])

                    calibration_mx = calibs[(sub, seq, cam)]
                    fx.extend([calibration_mx[0, 0]] * num_frames)
                    fy.extend([calibration_mx[1, 1]] * num_frames)
                    cx.extend([calibration_mx[0, 2]] * num_frames)
                    cy.extend([calibration_mx[1, 2]] * num_frames)

        self.pose2d_jointset = CocoExJoints()
        self.pose3d_jointset = MuPoTSJoints()

        self.poses2d = np.concatenate(poses2d)
        self.poses3d = np.concatenate(poses3d)
        self.valid_2d_pred = np.concatenate(valid_2d_pred)
        self.index = np.rec.array(index, dtype=[('seq', 'U12'), ('sub', 'int32'), ('subseq', 'int32'),
                                                ('cam', 'int32'), ('frame', 'int32')])

        self.fx = np.array(fx, dtype='float32')
        self.fy = np.array(fy, dtype='float32')
        self.cx = np.array(cx, dtype='float32')
        self.cy = np.array(cy, dtype='float32')

        self.sequences = sorted(sequences)

        assert len(self.poses2d) == len(self.index), len(self.index)

        assert len(self.poses2d) == len(self.poses3d)
        assert len(self.poses2d) == len(self.index), len(self.index)
        assert len(self.poses2d) == len(self.valid_2d_pred), len(self.valid_2d_pred)
        assert len(self.poses2d) == len(self.fx), len(self.fx)
        assert len(self.poses2d) == len(self.fy), len(self.fy)
        assert len(self.poses2d) == len(self.cx), len(self.cx)
        assert len(self.poses2d) == len(self.cy), len(self.cy)

    def filter_dataset(self, inds):
        super().filter_dataset(inds)
        self.sequences = sorted(np.unique(self.index.seq))

    def prepare_sample(self, ind):
        if isinstance(ind, (list, tuple, np.ndarray)):
            width = np.full(len(ind), 2048, dtype='int32')
        else:
            width = 2048

        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind],
                  'index': ind, 'valid_pose': self.valid_2d_pred[ind], 'cx': self.cx[ind], 'width': width}

        return sample


class PersonStackedMucoTempDataset(FlippableDataset, TemporalAugmentMixin):
    """ This dataset contains Muco-Temp poses, poses on the same frame are separated. """

    def __init__(self, pose2d_type, pose3d_scaling):
        assert pose2d_type == 'hrnet', "only hrnet is implemented"
        assert pose3d_scaling in ['univ', 'normal']

        self.transform = None

        self.pose2d_jointset = PersonStackedMuPoTsDataset.get_jointset(pose2d_type)
        self.pose3d_jointset = MuPoTSJoints()

        pose3d_key = 'annot3' if pose3d_scaling == 'normal' else 'univ_annot3'

        poses2d = []
        poses3d = []
        valid_2d_pred = []  # True if HR-net found a pose
        fx = []
        fy = []
        cx = []
        cy = []
        index = []

        calibs = mpii_3dhp.get_calibration_matrices()
        meta_data = muco_temp.get_metadata()

        for cam in range(11):
            gt = muco_temp.load_gt(cam)

            for vid in range(7):
                orig_shape = gt[vid][pose3d_key].shape  # (nFrames, nPoses, nJoints, 3)
                poses3d.append(_column_stack(gt[vid][pose3d_key]))

                kp = muco_temp.load_hrnet(cam, vid)
                poses2d.append(_column_stack(kp['poses']))
                valid_2d_pred.append(_column_stack(kp['is_valid']))

                assert len(poses3d[-1]) == len(poses2d[-1]), \
                    "Gt and predicted frames are not aligned, cam:" + str(cam)

                orig_frame = np.tile(np.arange(orig_shape[0]).reshape(-1, 1), (1, orig_shape[1]))
                orig_pose = np.tile(np.arange(orig_shape[1]).reshape(1, -1), (orig_shape[0], 1))
                orig_frame = _column_stack(orig_frame)  # (nFrames*nPoses,)
                orig_pose = _column_stack(orig_pose)

                index.extend([('%d/%d/%d' % (cam, vid, orig_pose[i]), cam, vid, orig_frame[i], orig_pose[i])
                              for i in range(len(orig_frame))])

                for pose_ind in range(orig_shape[1]):
                    sub, seq, _ = meta_data[cam][vid][pose_ind]
                    calibration_mx = calibs[(sub, seq, cam)]
                    fx.extend([calibration_mx[0, 0]] * orig_shape[0])
                    fy.extend([calibration_mx[1, 1]] * orig_shape[0])
                    cx.extend([calibration_mx[0, 2]] * orig_shape[0])
                    cy.extend([calibration_mx[1, 2]] * orig_shape[0])

        self.poses2d = np.concatenate(poses2d)
        self.poses3d = np.concatenate(poses3d)
        self.valid_2d_pred = np.concatenate(valid_2d_pred)
        self.index = np.rec.array(index, dtype=[('seq', 'U12'), ('cam', 'int32'), ('vid', 'int32'),
                                                ('frame', 'int32'), ('pose', 'int32')])

        self.fx = np.array(fx, dtype='float32')
        self.fy = np.array(fy, dtype='float32')
        self.cx = np.array(cx, dtype='float32')
        self.cy = np.array(cy, dtype='float32')

        assert len(self.poses2d) == len(self.index), len(self.index)

        assert len(self.poses2d) == len(self.poses3d)
        assert len(self.poses2d) == len(self.index), len(self.index)
        assert len(self.poses2d) == len(self.valid_2d_pred), len(self.valid_2d_pred)
        assert len(self.poses2d) == len(self.fx), len(self.fx)
        assert len(self.poses2d) == len(self.fy), len(self.fy)
        assert len(self.poses2d) == len(self.cx), len(self.cx)
        assert len(self.poses2d) == len(self.cy), len(self.cy)

    def prepare_sample(self, ind):
        if isinstance(ind, (list, tuple, np.ndarray)):
            width = np.full(len(ind), 2048, dtype='int32')
        else:
            width = 2048

        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind],
                  'index': ind, 'valid_pose': self.valid_2d_pred[ind], 'cx': self.cx[ind], 'width': width}

        return sample


def pose_grid_from_index(keys):
    """
    From an array of frame ids returns the id of the frame and the pose in that frame.
    These can be used to reshape arrays containing stacked poses.

    Parameters:
        keys: ids of frames. It is expected that poses in the same frame are next to each other (in other words,
                if keys[i]==keys[j], then for all i<=k<=j keys[k]==keys[i])
    """
    different = keys[1:] != keys[:-1]
    different = np.concatenate([[True], different])  # True if the current record is on a new frame compared to the previous record

    # frame_start will hold the index of the first pose of the current frame
    frame_start = -np.ones(len(different), dtype='int64')
    frame_start[different] = np.arange(len(different))[different]  # if different True, then contains the index, otherwise a -1
    # frame_start[i]==-1 if it is on the same frame as the previous pose, so max copies the prev value
    frame_start = np.maximum.accumulate(frame_start)
    pose_ind = np.arange(len(different)) - frame_start

    return keys, pose_ind


def reshape_posearray(frame_ind, pose_ind, array):
    """
    Reshapes an array that is aligned with a stacked pose array into one
    that is aligned to a by-frame grouped array. Unused places in the output are nan-ed out for
    floating point types. Other types are kept.

    NOTE: uses hardcoded number of poses, as in this project max people on a frame is 6.
    """
    assert np.max(pose_ind) < 6, "In this code the maximum number of poses per frame is hardcoded to 6"

    num_frames = np.max(frame_ind) + 1
    shape = (num_frames, 6) + array.shape[1:]

    result = np.zeros(shape, dtype=array.dtype)
    if array.dtype == 'float32' or array.dtype == 'float64':
        result = result * np.nan
    elif isinstance(array, np.recarray):
        result = np.rec.array(result)

    result[frame_ind, pose_ind] = array
    return result
