""" Generates Mask-RCNN bounding boxes. """
import argparse

from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import torch
import os
from torch.utils.data import DataLoader, Dataset
from util.misc import save




def get_kpdetection_conf():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

    return cfg


def get_model(cfg):
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return model


class ImgDirDataset(Dataset):

    def __init__(self, folder, transform):
        self.folder = folder
        self.files = sorted(os.listdir(folder))

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind):
        img = cv2.imread(os.path.join(self.folder, self.files[ind]))
        height, width = img.shape[:2]
        image = self.transform.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).contiguous()

        return {"image": image, "height": height, "width": width, "name": self.files[ind]}


def predict_dataset(model, dataset, out_folder, batch_size=16):

    loader = DataLoader(dataset, batch_size, collate_fn=lambda x: x, num_workers=3)

    with torch.no_grad():
        for batch in loader:
            predictions = model(batch)

            for i in range(len(batch)):
                boxes = predictions[i]['instances'].pred_boxes.tensor.cpu().numpy()
                scores = predictions[i]['instances'].scores.cpu().numpy()[:, np.newaxis]

                output = np.concatenate([boxes, scores], axis=1)
                assert output.shape[1] == 5
                save(os.path.join(out_folder, "%s.pkl" % batch[i]['name']), output)


def predict_imgs(input_path, output_path):
    cfg = get_kpdetection_conf()
    model = get_model(cfg)
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    assert cfg.INPUT.FORMAT == 'BGR'

    dataset = ImgDirDataset(input_path, transform_gen)
    predict_dataset(model, dataset, output_path, batch_size=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help="the path to the input frames")
    parser.add_argument('output_path', help="bboxes will be generated here")
    args = parser.parse_args()

    predict_imgs(args.input_path, args.output_path)
