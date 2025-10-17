import torch
from PIL import Image
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from torchvision.ops import box_convert


class InstanceIoU:
    def __init__(self, model_config_path, model_checkpoint_path, device="cuda"):
        self.gdino = load_model(model_config_path, model_checkpoint_path)
        self.gdino.to(device)
        self.box_threshold = 0.35
        self.text_threshold = 0.25

    def set_image(self, image_path):
        self.image_source, self.image = load_image(image_path)

    def get_bbox_from_instance(self, instance_label):
        boxes, logits, phrases = predict(
            model=self.gdino,
            image=self.image,
            caption=instance_label,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return boxes, logits

    def compute_iou_with_gdinoboxes(self, gtbox, gdinoboxes):
        iou = 0.0
        idx = 0
        for i in range(gdinoboxes.shape[0]):
            now_iou = self.compute_iou(gtbox, gdinoboxes[i])
            if now_iou > iou:
                iou = now_iou
                idx = i
        return iou, idx

    def compute_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        x_left = max(x1, x1_)
        y_top = max(y1, y1_)
        x_right = min(x2, x2_)
        y_bottom = min(y2, y2_)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_ - x1_) * (y2_ - y1_)

        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
