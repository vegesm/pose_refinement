"""
generates the muco_temp synthetic dataset. In order to use this script, you already have to have to have generated
the sequence meta data files in 'sequence_meta.pkl' and the ground-truth poses. The scripts can be found in mpi_inf_3dhp.ipynb
"""
from databases import mpii_3dhp, muco_temp
from databases.joint_sets import MuPoTSJoints
import numpy as np
from util.misc import ensuredir, load
import os
import cv2
from multiprocessing import Pool

NUM_FRAMES = 2000


def generate_vid_frames(cam, vid_id):
    print(cam, vid_id)
    metas = sequence_metas[cam][vid_id]
    steps = [2 if mpii_3dhp.get_train_fps(meta[0], meta[1]) == 50 else 1 for meta in metas]
    out_folder = os.path.join(muco_temp.MUCO_TEMP_PATH, 'frames/cam_%d/vid_%d' % (cam, vid_id))
    ensuredir(out_folder)

    gt_poses = load(os.path.join(muco_temp.MUCO_TEMP_PATH, 'frames/cam_%d/gt.pkl' % cam))[vid_id]['annot3']
    hip_ind = MuPoTSJoints().index_of('hip')

    for i in range(NUM_FRAMES):
        # generate frame
        depths = gt_poses[i, :, hip_ind, 2]
        ordered_poses = np.argsort(depths)[::-1]  # poses ordered by depth in decreasing order

        bg_ind = ordered_poses[0]
        img = mpii_3dhp.get_image(metas[bg_ind][0], metas[bg_ind][1], cam, metas[bg_ind][2] + i * steps[bg_ind], rgb=False)
        img = img.astype('float32')
        # add new pose onto image
        for pose_ind in ordered_poses[1:]:
            sub, seq, start = metas[pose_ind]
            pose_img = mpii_3dhp.get_image(sub, seq, cam, start + i * steps[pose_ind], rgb=False)

            # mask is 0 at greenscreen bg, 1 at foreground (body, chair)
            mask = mpii_3dhp.get_mask(sub, seq, cam, start + i * steps[pose_ind], 'FGmasks')[:, :, 2] / 255.
            mask = cv2.GaussianBlur(mask, (0, 0), 2)[:, :, np.newaxis]
            # chair_mask is 0 at chair, 1 everywhere else
            chair_mask = mpii_3dhp.get_mask(sub, seq, cam, start + i * steps[pose_ind], 'ChairMasks')[:, :, [2]] / 255

            img = chair_mask * img + (1 - chair_mask) * pose_img
            img = mask * pose_img + (1 - mask) * img

        img = img.astype('uint8')
        cv2.imwrite(os.path.join(out_folder, 'img_%04d.jpg' % i), img, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    sequence_metas = muco_temp.get_metadata()
    p = Pool(6)
    params = [(cam, vid) for cam in range(11) for vid in range(0, 7)]
    p.starmap(generate_vid_frames, params)
