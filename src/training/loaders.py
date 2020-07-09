import numpy as np

from torch.utils.data import DataLoader, SequentialSampler
from itertools import chain
import torch

from databases.datasets import pose_grid_from_index, Mpi3dTrainDataset, PersonStackedMucoTempDataset, ConcatPoseDataset


class ConcatSampler(torch.utils.data.Sampler):
    """ Concatenates two samplers. """

    def __init__(self, sampler1, sampler2):
        self.sampler1 = sampler1
        self.sampler2 = sampler2

    def __iter__(self):
        return chain(iter(self.sampler1), iter(self.sampler2))

    def __len__(self):
        return len(self.sampler1) + len(self.sampler2)



class UnchunkedGenerator:
    """
    Loader that can be used with VideoPose3d model to load all frames of a video at once.
    Useful for testing/prediction.
    """
    def __init__(self, dataset, pad, augment):
        self.seqs = sorted(np.unique(dataset.index.seq))
        self.dataset = dataset
        self.pad = pad
        self.augment = augment

    def __iter__(self):
        for seq in self.seqs:
            inds = np.where(self.dataset.index.seq == seq)[0]
            batch = self.dataset.get_samples(inds, False)
            batch_2d = np.expand_dims(np.pad(batch['pose2d'], ((self.pad, self.pad), (0, 0)), 'edge'), axis=0)
            batch_3d = np.expand_dims(batch['pose3d'], axis=0)
            batch_valid = np.expand_dims(batch['valid_pose'], axis=0)

            if self.augment:
                flipped_batch = self.dataset.get_samples(inds, True)
                flipped_batch_2d = np.expand_dims(np.pad(flipped_batch['pose2d'],
                                                         ((self.pad, self.pad), (0, 0)), 'edge'), axis=0)
                flipped_batch_3d = np.expand_dims(flipped_batch['pose3d'], axis=0)

                batch_2d = np.concatenate((batch_2d, flipped_batch_2d), axis=0)
                batch_3d = np.concatenate((batch_3d, flipped_batch_3d), axis=0)
                batch_valid = np.concatenate((batch_valid, batch_valid), axis=0)

            #             yield {'pose2d': batch_2d, 'pose3d':batch_3d}
            yield batch_2d, batch_valid




class ChunkedGenerator:
    """
    Generator to be used with temporal model, during training.
    """

    def __init__(self, dataset, batch_size, pad, augment, shuffle=True):
        """
        pad: 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
               it is usually (receptive_field-1)/2
        augment: turn on random horizontal flipping for training
        shuffle: randomly shuffle the dataset before each epoch
        """
        assert isinstance(dataset, (Mpi3dTrainDataset, PersonStackedMucoTempDataset, ConcatPoseDataset)), "Only works with Mpi datasets"
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad = pad
        self.shuffle = shuffle
        self.augment = augment

        N = len(dataset.index)
        frame_start = np.arange(N)-pose_grid_from_index(dataset.index.seq)[1]  # index of the start of the frame
        frame_end = np.arange(N)-pose_grid_from_index(dataset.index.seq[::-1])[1]
        frame_end = N-frame_end[::-1]-1   # index of the end of the frame (last frame)

        self.frame_start = frame_start
        self.frame_end = frame_end

        assert np.all(frame_start<=frame_end)
        assert np.all(dataset.index.seq[frame_start] == dataset.index.seq[frame_end])
        assert np.all(dataset.index.seq[frame_start] == dataset.index.seq)

    def __len__(self):
        return len(self.dataset)//self.batch_size

    def __iter__(self):
        N = len(self.dataset)
        num_batch = N//self.batch_size

        indices = np.arange(N)
        if self.shuffle:
            np.random.shuffle(indices)

        SUB_BATCH = 4
        assert self.batch_size % SUB_BATCH == 0, "SUB_BATCH must divide batch_size"

        class LoadingDataset:
            def __len__(iself):
                return num_batch*SUB_BATCH

            def __getitem__(iself, ind):
                sub_batch_size = self.batch_size//SUB_BATCH
                batch_inds = indices[ind*sub_batch_size: (ind+1)*sub_batch_size]  # (nBatch,)
                batch_frame_start = self.frame_start[batch_inds][:, np.newaxis]
                batch_frame_end = self.frame_end[batch_inds][:, np.newaxis]

                if self.augment:
                    flip = np.random.random(sub_batch_size) < 0.5
                else:
                    flip = np.zeros(sub_batch_size, dtype='bool')
                flip = np.tile(flip[:, np.newaxis], (1, 2*self.pad+1))

                # expand batch_inds such that it includes lower&upper bound indices for every element
                chunk_inds = batch_inds[:, np.newaxis] + np.arange(-self.pad, self.pad+1)[np.newaxis, :]
                chunk_inds = np.clip(chunk_inds, batch_frame_start, batch_frame_end)
                assert np.all(chunk_inds>=batch_frame_start)
                assert np.all(chunk_inds<=batch_frame_end)

                chunk = self.dataset.get_samples(chunk_inds.ravel(), flip.ravel())
                chunk_pose2d = chunk['pose2d'].reshape(chunk_inds.shape+chunk['pose2d'].shape[1:])
                chunk_pose3d = chunk['pose3d'].reshape(chunk_inds.shape+chunk['pose3d'].shape[1:])
                chunk_valid = chunk['valid_pose'].reshape(chunk_inds.shape+chunk['valid_pose'].shape[1:])
                # for non temporal values select the middle item:
                chunk_pose3d = chunk_pose3d[:, self.pad]
                chunk_valid = chunk_valid[:, self.pad]

                chunk_pose3d = np.expand_dims(chunk_pose3d, 1)

                return chunk_pose2d, chunk_pose3d, chunk_valid

        wrapper_dataset = LoadingDataset()
        loader = DataLoader(wrapper_dataset, sampler=SequentialSampler(wrapper_dataset), 
                            batch_size=SUB_BATCH, num_workers=4)

        for chunk_pose2d, chunk_pose3d, chunk_valid in loader:
            chunk_pose2d = chunk_pose2d.reshape((-1,)+chunk_pose2d.shape[2:])
            chunk_pose3d = chunk_pose3d.reshape((-1,)+chunk_pose3d.shape[2:])
            chunk_valid = chunk_valid.reshape(-1)
            yield {'temporal_pose2d': chunk_pose2d, 'pose3d': chunk_pose3d, 'valid_pose': chunk_valid}
