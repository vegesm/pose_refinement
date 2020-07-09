import argparse
import os

from databases.datasets import Mpi3dTestDataset, Mpi3dTrainDataset, PersonStackedMucoTempDataset, ConcatPoseDataset
from model.videopose import TemporalModel, TemporalModelOptimized1f
from training.callbacks import preds_from_logger, ModelCopyTemporalEvaluator
from training.loaders import ChunkedGenerator
from training.preprocess import *
from training.torch_tools import torch_train
from util.misc import save, ensuredir


def calc_loss(model, batch, config):
    if config['model']['loss'] == 'l1_nan':
        pose2d = batch['temporal_pose2d']
        gt_3d = batch['pose3d']

        # different handling for numpy and PyTorch inputs
        if isinstance(pose2d, torch.Tensor):
            inds = torch.all(torch.all(1 - (pose2d != pose2d), dim=(-1)), dim=-1)
            pose2d = pose2d[inds]
            gt_3d = gt_3d[inds]
            pose2d = pose2d.to('cuda')
            gt_3d = gt_3d.to('cuda')
        else:
            inds = np.all(~np.isnan(pose2d), axis=(-1, -2))
            pose2d = pose2d[inds]
            gt_3d = gt_3d[inds]
            pose2d = torch.from_numpy(pose2d).to('cuda')
            gt_3d = torch.from_numpy(gt_3d).to('cuda')

    elif config['model']['loss'] == 'l1':
        pose2d = batch['temporal_pose2d']
        gt_3d = batch['pose3d']
        pose2d = pose2d.to('cuda')
        gt_3d = gt_3d.to('cuda')

    # forward pass
    pred_3d = model(pose2d)

    if config['model']['loss'] == 'l1':
        loss_3d = torch.nn.functional.l1_loss(pred_3d, gt_3d)
    elif config['model']['loss'] == 'l1_nan':
        loss_3d = torch.nn.functional.l1_loss(pred_3d, gt_3d)
    else:
        raise Exception('Unknown pose loss: ' + str(config['model']['loss']))

    return loss_3d, {'loss_3d': loss_3d.item()}


def run_experiment(output_path, _config):
    save(os.path.join(output_path , 'config.json'), _config)
    ensuredir(output_path)

    if _config['train_data'] == 'mpii_train':
        print("Training data is mpii-train")
        train_data = Mpi3dTrainDataset(_config['pose2d_type'], _config['pose3d_scaling'],
                                       _config['cap_25fps'], _config['stride'])

    elif _config['train_data'] == 'mpii+muco':
        print("Training data is mpii-train and muco_temp concatenated")
        mpi_data = Mpi3dTrainDataset(_config['pose2d_type'], _config['pose3d_scaling'],
                                     _config['cap_25fps'], _config['stride'])

        muco_data = PersonStackedMucoTempDataset(_config['pose2d_type'], _config['pose3d_scaling'])
        train_data = ConcatPoseDataset(mpi_data, muco_data)

    elif _config['train_data'].startswith('muco_temp'):
        train_data = PersonStackedMucoTempDataset(_config['pose2d_type'], _config['pose3d_scaling'])

    test_data = Mpi3dTestDataset(_config['pose2d_type'], _config['pose3d_scaling'], eval_frames_only=True)

    if _config['simple_aug']:
        train_data.augment(False)

    # Load the preprocessing steps
    train_data.transform = None
    transforms_train = [decode_trfrm(_config['preprocess_2d'], globals())(train_data, cache=False),
                        decode_trfrm(_config['preprocess_3d'], globals())(train_data, cache=False)]

    normalizer2d = transforms_train[0].normalizer
    normalizer3d = transforms_train[1].normalizer

    transforms_test = [decode_trfrm(_config['preprocess_2d'], globals())(test_data, normalizer2d),
                       decode_trfrm(_config['preprocess_3d'], globals())(test_data, normalizer3d)]

    transforms_train.append(RemoveIndex())
    transforms_test.append(RemoveIndex())

    train_data.transform = SaveableCompose(transforms_train)
    test_data.transform = SaveableCompose(transforms_test)

    # save normalisation params
    save(output_path+'/preprocess_params.pkl', train_data.transform.state_dict())

    print("Length of training data:", len(train_data))
    print("Length of test data:", len(test_data))

    model = TemporalModelOptimized1f(train_data[[0]]['pose2d'].shape[-1],
                                     MuPoTSJoints.NUM_JOINTS, _config['model']['filter_widths'],
                                     dropout=_config['model']['dropout'], channels=_config['model']['channels'],
                                     layernorm=_config['model']['layernorm'])
    test_model = TemporalModel(train_data[[0]]['pose2d'].shape[-1],
                               MuPoTSJoints.NUM_JOINTS, _config['model']['filter_widths'],
                               dropout=_config['model']['dropout'], channels=_config['model']['channels'],
                               layernorm=_config['model']['layernorm'])

    model.cuda()
    test_model.cuda()

    save(output_path+'/model_summary.txt', str(model))

    pad = (model.receptive_field() - 1) // 2
    train_loader = ChunkedGenerator(train_data, _config['batch_size'], pad, _config['train_time_flip'], shuffle=True)
    tester = ModelCopyTemporalEvaluator(test_model, test_data, _config['model']['loss'], _config['test_time_flip'],
                                        post_process3d=get_postprocessor(_config, test_data, normalizer3d), prefix='test')

    torch_train(train_loader, model, lambda m, b: calc_loss(m, b, _config), _config, callbacks=[tester])

    torch.save(model.state_dict(), os.path.join(output_path, 'model_params.pkl'))
    save(output_path+'/test_results.pkl', {'index': test_data.index, 'pred': preds_from_logger(test_data, tester),
                                     'pose3d': test_data.poses3d})


def main(output_path):
    params = {
        'num_epochs': 80,
        'preprocess_2d': 'DepthposeNormalize2D',
        'preprocess_3d': 'SplitToRelativeAbsAndMeanNormalize3D',

        # training
        'optimiser': 'adam',
        'adam_amsgrad': True,
        'learning_rate': 1e-3,
        'sgd_momentum': 0,
        'batch_size': 1024,
        'train_time_flip': True,
        'test_time_flip': True,

        'lr_scheduler': {
            'type': 'multiplicative',
            'multiplier': 0.95,
            'step_size': 1,
        },

        # dataset
        'train_data': 'mpii_train',
        'pose2d_type': 'hrnet',
        'pose3d_scaling': 'normal',
        'megadepth_type': 'megadepth_at_hrnet',
        'cap_25fps': True,
        'stride': 2,
        'simple_aug': True,  # augments data by duplicating each frame

        'model': {
            'loss': 'l1',
            'channels': 1024,
            'dropout': 0.25,
            'filter_widths': [3, 3, 3, 3],
            'layernorm': False,
        },
    }

    run_experiment(output_path, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='../output', help='folder to save the model to')
    args = parser.parse_args()

    main(args.output)
