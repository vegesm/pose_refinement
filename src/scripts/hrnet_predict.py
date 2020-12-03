from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append('../hrnet/lib')

from scripts import hrnet_dataset

# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Marton Veges
# ------------------------------------------------------------------------------

import argparse
import time
import os

import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np

from config import cfg
from config import update_config
from core.function import AverageMeter
from utils.utils import create_logger
from core.inference import get_final_preds
from utils.transforms import flip_back

import models

from util.misc import load, ensuredir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('path', help="the path to the video frames and bboxes", type=str)

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        default='../hrnet/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    return parser.parse_args()


def predict_dataset(config, dataset, model):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_names = []
    orig_boxes = []

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        pin_memory=True, num_workers=1
    )

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, meta) in enumerate(data_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            names = meta['image']
            image_names.extend(names)
            orig_boxes.extend(meta['origbox'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time)
                print(msg)

        return all_preds, all_boxes, image_names, orig_boxes


def predict_imgs(model, img_folder, bbox_folder, output_file, normalize, detection_thresh):
    detections = {}

    for file in sorted(os.listdir(bbox_folder)):
        dets = load(os.path.join(bbox_folder, file))
        assert dets.shape[1] == 5
        img_name = file[:-4]  # remove extension
        detections[img_name] = dets

    valid_dataset = hrnet_dataset.ImgFolderDataset(cfg, img_folder, detections,
                                                   normalize, detection_thresh)

    start = time.time()
    preds, boxes, image_names, orig_boxes = predict_dataset(cfg, valid_dataset, model)
    end = time.time()
    print("Time in prediction: " + str(end - start))

    ensuredir(os.path.dirname(output_file))
    valid_dataset.rescore_and_save_result(output_file, preds, boxes, image_names, orig_boxes)


def predict(cfg_path, img_dir, bbox_dir, out_file, param_overrides=[]):
    # update_config needs some hardcoded params, fake them here
    class args:
        cfg = cfg_path
        opts = param_overrides
        modelDir = ''
        logDir = ''
        dataDir = ''

    update_config(cfg, args)
    cfg.defrost()
    cfg.TEST.MODEL_FILE = '../hrnet/pose_hrnet_w32_256x192.pth'
    cfg.TEST.USE_GT_BBOX = False
    cfg.TEST.BATCH_SIZE_PER_GPU = 64
    cfg.GPUS = (0,)
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(cfg, cfg_path, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Compose([transforms.ToTensor(), normalize])

    detection_thresh = 0.8

    img_dir = os.path.join(img_dir, '*')  # Dataset requires a glob format
    predict_imgs(model, img_dir, bbox_dir, out_file, normalize, detection_thresh)


if __name__ == '__main__':
    args = parse_args()

    img_dir = os.path.join(args.path, 'frames')
    bbox_dir = os.path.join(args.path, 'bboxes')
    out_file = os.path.join(args.path, 'keypoints.json')

    predict(args.cfg, img_dir, bbox_dir, out_file, param_overrides=args.opts)