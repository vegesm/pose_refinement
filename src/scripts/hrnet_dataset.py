# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Marton Veges
# ------------------------------------------------------------------------------



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import logging
import os
from util.misc import ensuredir

import json_tricks as json
import numpy as np

import copy
import cv2
import glob
from torch.utils.data import Dataset
from utils.transforms import get_affine_transform

from nms.nms import oks_nms
from nms.nms import soft_oks_nms

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    A DataLoader loading bounding boxes for CoCo joints evaluation.

        "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    """

    def __init__(self, cfg):
        # Unpack NMS threshold parameters
        self.image_thre = cfg.TEST.IMAGE_THRE  # bounding boxes lower than this value are not predicted, just thrown away
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE

        # Unpack image size parameters
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

    def _lurb2cs(self, box):  # TODO check!!!!!1111
        x, y = box[:2]
        w = box[2] - box[0]
        h = box[3] - box[1]
        return self._xywh2cs(x, y, w, h)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def rescore_and_save_result(self, output_file, preds, all_boxes, img_path, orig_boxes):
        assert output_file.endswith('.json') or output_file.endswith('.npy'), "Only json and numpy output is supported"

        ensuredir(os.path.dirname(output_file))

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': img_path[idx],
                'origbox': orig_boxes[idx]
            })

        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        nmsed_kpts_by_frame = defaultdict(list)
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
            else:
                keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)

            if len(keep) == 0:
                selected_kpts = img_kpts
            else:
                selected_kpts = [img_kpts[_keep] for _keep in keep]

            oks_nmsed_kpts.append(selected_kpts)
            nmsed_kpts_by_frame[img] = selected_kpts

        self._write_keypoint_results(nmsed_kpts_by_frame, output_file)

    def _write_keypoint_results(self, keypoints, output_file):
        # TODO turn list into numpy arrays
        if output_file.endswith('.json'):
            # Convert numpy arrays to Python lists
            for img_name, poses in keypoints.items():
                for pose in poses:
                    pose['center'] = pose['center'].tolist()
                    pose['scale'] = pose['scale'].tolist()
                    pose['keypoints'] = pose['keypoints'].ravel().tolist()
                    pose['origbox'] = pose['origbox'].tolist()

            with open(output_file, 'w') as f:
                json.dump(keypoints, f, sort_keys=True, indent=4)

        elif output_file.endswith('npy'):
            frame_ind = keypoints.keys()
            assert all([f.startswith('videocap#') for f in frame_ind])
            frame_ind = sorted(frame_ind, key=lambda x: int(x[len('videocap#'):]))

            kps = []
            for f in frame_ind:
                assert len(keypoints[f]) == 1, 'Only images with a single pose are supported in numpy save mode, found: ' + str(
                    keypoints[f])
                kps.append(keypoints[f][0]['keypoints'])

            kps = np.stack(kps, axis=0)
            print("shape:" + str(kps.shape))
            np.save(output_file, kps)
        else:
            raise NotImplementedError('Unknown file ending: ' + output_file)


class ImgFolderDataset(BaseDataset):
    """ Can be used with a folder of images"""

    def __init__(self, cfg, img_path, dets, transform, det_threshold):
        """

        :param cfg: config object
        :param img_path: path to folder, must be glob (e.g. *.jpg)
        :param dets: detections img->boxes
        :param transform: transformations to apply on images
        """
        super(ImgFolderDataset, self).__init__(cfg)

        self.img_paths = sorted(glob.glob(img_path))
        self.basedir = os.path.dirname(img_path)
        self.dets = dets
        self.transform = transform
        self.image_thre = det_threshold

        # check there is a file for all detections
        # img_names = set([os.path.basename(x) for x in self.img_paths])
        # for img in dets:
        #     assert img in img_names, "Could not find " + img

        self.db = self._prepare_db()

        self.last_idx_read = None
        self.last_img_read = None
        self.last_img = None

    def _prepare_db(self):
        """
        Prepares the detections from the self.dets field. Optionally filters out detected bounding boxes if their
        score is low.
        """
        kpt_db = []

        filtered_boxes_num = 0
        total_boxes_num = 0
        for img_name in sorted(self.dets.keys()):
            boxes = self.dets[img_name]
            total_boxes_num += len(boxes)

            for box in boxes:
                score = box[4]
                if score < self.image_thre:
                    continue

                filtered_boxes_num = filtered_boxes_num + 1

                center, scale = self._lurb2cs(box[:4])
                kpt_db.append({
                    'image': img_name,
                    'center': center,
                    'scale': scale,
                    'score': score,
                    'origbox': box[:4]
                })

        logger.info('=> Total boxes: {}'.format(total_boxes_num))
        logger.info('=> Total boxes after filter low score@{}: {}'.format(self.image_thre, filtered_boxes_num))
        return kpt_db

    def _get_img(self, img_name):
        # Read from cache
        if self.last_img_read == img_name:
            return self.last_img

        # assert self.last_img_read is None or self.last_frame_read == idx - 1, "Can only read sequentially %d -> %d" % \
        #                                                                         (self.last_frame_read, idx)

        img = cv2.imread(os.path.join(self.basedir, img_name))
        assert img is not None, "could not find " + img_name

        if self.color_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.last_img_read = img_name
        self.last_img = img
        return img

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        assert self.last_idx_read is None or self.last_idx_read == idx - 1, "idx junmp: %d -> %d" % (self.last_idx_read, idx)
        db_rec = copy.deepcopy(self.db[idx])
        self.last_idx_read = idx

        image_file = db_rec['image']

        frame = self._get_img(image_file)

        if frame is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(frame, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        meta = {
            'image': image_file,
            'origbox': db_rec['origbox'],
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, meta
