import argparse

import cv2
import numpy as np
import os
from databases.datasets import FlippableDataset
from databases.joint_sets import MuPoTSJoints, CocoExJoints
from model.pose_refinement import optimize_poses
from scripts.eval import load_model, LOG_PATH
from training.callbacks import TemporalTestEvaluator
from training.preprocess import SaveableCompose, MeanNormalize3D, get_postprocessor
from util.misc import load, save, ensuredir
from util.pose import extend_hrnet_raw
import shutil
import sys
from scripts import maskrcnn_bboxes, hrnet_predict


def stack_hrnet_raw(img_names, keypoints_raw):
    """
    Gets a JSON output of hrnet and converts it into a numpy array. Only works
    with single pose images.

    Parameters:
        img_names: an array of image names to determine the output order

    Returns:
        keypoints: ndarray(len(img_names), 19, 3) - the 2D keypoints according to HR-net. Invalid poses have all-zero values
        is_valid: ndarray(len(img_names),) - True if a person was detected on the image
    """
    keypoints = []
    is_valid = []
    for frame in img_names:
        if frame in keypoints_raw:
            scores = [x['score'] for x in keypoints_raw[frame]]
            best_ind = np.argmax(scores)

            pose = np.array(keypoints_raw[frame][best_ind]['keypoints'])
            pose = pose.reshape((17, 3))
            is_valid.append(True)
        else:
            pose = np.zeros((17, 3))  # placeholder value
            is_valid.append(False)

        keypoints.append(pose)

    keypoints = np.stack(keypoints)
    keypoints = extend_hrnet_raw(keypoints).astype('float32')
    is_valid = np.array(is_valid)
    keypoints[~is_valid] = 0

    return keypoints, is_valid


class VideoTemporalDataset(FlippableDataset):

    def __init__(self, frame_folder, hrnet_keypoint_file, fx, fy, cx=None, cy=None):
        self.transform = None

        self.pose2d_jointset = CocoExJoints()
        self.pose3d_jointset = MuPoTSJoints()

        frame_list = sorted(os.listdir(frame_folder))
        N = len(frame_list)

        hrnet_detections = load(hrnet_keypoint_file)
        self.poses2d, self.valid_2d_pred = stack_hrnet_raw(frame_list, hrnet_detections)
        assert len(self.poses2d) == N, "unexpected number of frames"

        index = [('vid', i) for i in range(N)]
        self.index = np.rec.array(index, dtype=[('seq', 'U4'), ('frame', 'int32')])

        self.poses3d = np.ones((N, self.pose3d_jointset.NUM_JOINTS, 3))  # dummy values

        # load first frame to get width/height
        frame = cv2.imread(os.path.join(frame_folder, frame_list[0]))
        self.width = frame.shape[1]

        self.fx = np.full(N, fx, dtype='float32')
        self.fy = np.full(N, fy, dtype='float32')
        self.cx = np.full(N, cx if cx is not None else frame.shape[1] / 2, dtype='float32')
        self.cy = np.full(N, cy if cy is not None else frame.shape[0] / 2, dtype='float32')

        assert self.poses2d.shape[1] == self.pose2d_jointset.NUM_JOINTS

    def prepare_sample(self, ind):
        if isinstance(ind, (list, tuple, np.ndarray)):
            width = np.full(len(ind), self.width, dtype='int32')
        else:
            width = self.width

        sample = {'pose2d': self.poses2d[ind], 'pose3d': self.poses3d[ind],
                  'index': ind, 'valid_pose': self.valid_2d_pred[ind], 'cx': self.cx[ind], 'width': width}

        return sample


def run_tpn(model_name, img_folder, hrnet_keypoint_file, pose_refine, focal_length, cx, cy):
    config, m = load_model(model_name)
    dataset = VideoTemporalDataset(img_folder, hrnet_keypoint_file, focal_length, focal_length, cx, cy)

    params_path = os.path.join(LOG_PATH, str(model_name), 'preprocess_params.pkl')
    transform = SaveableCompose.from_file(params_path, dataset, globals())
    dataset.transform = transform

    assert isinstance(transform.transforms[1].normalizer, MeanNormalize3D)
    normalizer3d = transform.transforms[1].normalizer

    post_process_func = get_postprocessor(config, dataset, normalizer3d)

    logger = TemporalTestEvaluator(m, dataset, config['model']['loss'], True, post_process3d=post_process_func)
    logger.eval(calculate_scale_free=False, verbose=False)

    poses = logger.preds['vid']

    if pose_refine:
        print("Refining poses...")
        refine_config = load('../models/pose_refine_config.json')
        poses = optimize_poses(poses, dataset, refine_config)

    return poses


def main(args):
    assert args.output.endswith('.pkl'), "Output file must be a pkl file"

    # Clear up temp folder if it exists
    if os.path.exists(args.tmp_folder) and not args.tpn_only:
        shutil.rmtree(args.tmp_folder)

    frame_dir = os.path.join(args.tmp_folder, 'frames')
    bbox_dir = os.path.join(args.tmp_folder, 'bboxes')
    keypoint_file = os.path.join(args.tmp_folder, 'keypoints.json')

    if not args.tpn_only:
        # split to frames:
        ensuredir(os.path.join(args.tmp_folder, 'frames'))
        out = os.system("ffmpeg -i %s -qscale:v 2 %s/frames/img_%%06d.jpg" % (args.vid_path, args.tmp_folder))
        if out != 0:
            print("could not split to frames, error code: " + str(out))
            sys.exit(1)

        # Mask-RCNN
        maskrcnn_bboxes.predict_imgs(frame_dir, bbox_dir)

        # hrnet
        hrnet_predict.predict('../hrnet/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml', frame_dir,
                              bbox_dir, keypoint_file)

    # Run TPN
    print("Running TPN...")
    poses = run_tpn(args.model, frame_dir, keypoint_file, args.pose_refine, args.focal_length, args.cx, args.cy)
    save(args.output, poses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_path', help="Path to the video file")
    parser.add_argument('output', help="Output .pkl file")
    parser.add_argument('-m', '--model', default='normal', help="Name of the model (either 'normal' or 'universal')")
    parser.add_argument('-r', '--pose-refine', action='store_true', help='Apply pose-refinement after TPN')
    parser.add_argument('-f', '--focal-length', default=1200, type=float, help='focal length of the camera')
    parser.add_argument('-cx', '--cx', default=None, type=float, help='horizontal centerpoint of camera')
    parser.add_argument('-cy', '--cy', default=None, type=float, help='vertical centerpoint of camera')
    parser.add_argument('--tmp-folder', default='../tmp', help="Path to a folder where temporary results are stored")
    parser.add_argument('--tpn-only', action='store_true', help='Run the TPN only. This requires the bounding boxes and keypoints' +
                                                                ' already generated in the temporary folder.')
    args = parser.parse_args()

    main(args)
