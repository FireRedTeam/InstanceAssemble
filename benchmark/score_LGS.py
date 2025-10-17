import os
import json
import glob
import argparse
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from utils import InstanceIoU

def evaluate_one_image(
    image_path: str,
    bboxes_xyxy_norm: List[List[float]],
    labels: List[str],
    descriptions: List[str],
    instance_iou: InstanceIoU,
    model,
    tokenizer,
    iou_thr: float = 0.5,
) -> Tuple[float, float, float, float]:
    ious, colors, textures, shapes = [], [], [], []

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    for box_norm, label, desc in zip(bboxes_xyxy_norm, labels, descriptions):
        # IoU
        instance_iou.set_image(image_path)
        gdino_boxes, _scores = instance_iou.get_bbox_from_instance(label)
        iou, idx = instance_iou.compute_iou_with_gdinoboxes(box_norm, gdino_boxes)
        iou = float(iou)
        ious.append(iou if iou >= iou_thr else 0.0)

        score_color = score_texture = score_shape = 0.0

        if iou >= iou_thr and label != desc and len(gdino_boxes) > 0:
            x1, y1, x2, y2 = gdino_boxes[idx]
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            crop = img.crop((x1, y1, x2, y2))

            def yes_no(q: str) -> bool:
                msgs = [{"role": "user", "content": [crop, q]}]
                ans = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, seed=42)
                return "yes" in str(ans).lower()

            q_color = (
                f'Is the subject in "{label}" in the image consistent with the color described '
                f'in the detailed description: "{desc}"? Strictly answer with "Yes" or "No". '
                f'If the color is not mentioned, answer "Yes".'
            )
            q_texture = (
                f'Is the subject in "{label}" in the image consistent with the texture described '
                f'in the detailed description: "{desc}"? Strictly answer with "Yes" or "No". '
                f'If the texture is not mentioned, answer "Yes".'
            )
            q_shape = (
                f'Is the subject in "{label}" in the image consistent with the shape described '
                f'in the detailed description: "{desc}"? Strictly answer with "Yes" or "No". '
                f'If the shape is not mentioned, answer "Yes".'
            )

            if yes_no(q_color):
                score_color = iou
            if yes_no(q_texture):
                score_texture = iou
            if yes_no(q_shape):
                score_shape = iou

        colors.append(score_color)
        textures.append(score_texture)
        shapes.append(score_shape)

    return float(np.sum(ious)), float(np.sum(colors)), float(np.sum(textures)), float(np.sum(shapes))


def score_instance_dir(
    img_dir: str,
    instance_iou: InstanceIoU,
    model,
    tokenizer,
    iou_thr: float,
    save_detailed: bool,
) -> None:
    dense_path = os.path.join(img_dir, "images_denselayout")
    save_dir = os.path.join(img_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_txt = os.path.join(save_dir, "lgs_score.txt")
    detailed_json = os.path.join(save_dir, "lgs_score_detailed.json")

    dataset_repo = "FireRedTeam/DenseLayout"
    test_dataset = load_dataset(dataset_repo, split="test")

    all_bbox_count = all_iou = all_color = all_texture = all_shape = 0.0
    detailed_res = {}

    for data in tqdm(test_dataset, desc=f"Scoring {img_dir}"):
        gen_img_path = os.path.join(dense_path, f"{data['id']}.jpg")
        if not os.path.exists(gen_img_path):
            continue

        H, W = data["height"], data["width"]
        bboxes_xyxy_norm = []
        for box in [cond["bbox"] for cond in data["annos"]]:
            x1, y1, x2, y2 = box
            bboxes_xyxy_norm.append([x1 / W, y1 / H, x2 / W, y2 / H])

        labels = [cond["category_name"] for cond in data["annos"]]
        descriptions = [cond["caption"] for cond in data["annos"]]

        s_iou, s_col, s_tex, s_sha = evaluate_one_image(
            gen_img_path,
            bboxes_xyxy_norm,
            labels,
            descriptions,
            instance_iou,
            model,
            tokenizer,
            iou_thr=iou_thr,
        )

        n_boxes = len(bboxes_xyxy_norm)
        all_bbox_count += n_boxes
        all_iou += s_iou
        all_color += s_col
        all_texture += s_tex
        all_shape += s_sha

        if save_detailed:
            detailed_res[f"{data['id']}.jpg"] = {
                "box_count": n_boxes,
                "sum_iou": s_iou,
                "sum_color": s_col,
                "sum_texture": s_tex,
                "sum_shape": s_sha,
            }

    if all_bbox_count == 0:
        print("No boxes evaluated. Check your paths.")
        return

    miou = all_iou / all_bbox_count
    mcolor = all_color / all_bbox_count
    mtexture = all_texture / all_bbox_count
    mshape = all_shape / all_bbox_count

    if save_detailed:
        with open(detailed_json, "w", encoding="utf-8") as f:
            json.dump(detailed_res, f, indent=2)
        print(f"Detailed results saved to: {detailed_json}")

    with open(save_txt, "w", encoding="utf-8") as f:
        print(f"dense instance score: {int(all_bbox_count)} boxes", file=f)
        print(f"\t - iou score: {miou}", file=f)
        print(f"\t - color score: {mcolor}", file=f)
        print(f"\t - texture score: {mtexture}", file=f)
        print(f"\t - shape score: {mshape}", file=f)

    print(f"[OK] Metrics written to: {save_txt}")
    print(miou, mcolor, mtexture, mshape)


def main():
    parser = argparse.ArgumentParser("Evaluate LGS on DenseLayout generations")
    parser.add_argument("--imgdir", required=True, help="output image dir, e.g. ./output/fluxdev")
    parser.add_argument("--groundingdino_config_path", default="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="your path to groundingdino config, default ./groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--groundingdino_checkpoint_path", default="./GroundingDINO/weights/groundingdino_swint_ogc.pth", help="your path to groundingdino checkpoint, default ./groundingdino/weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--vlm_id", default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--save_detailed", action="store_true", help="Save per-image sums to JSON")
    args = parser.parse_args()

    device = args.device
    torch_dtype = torch.bfloat16

    instance_iou = InstanceIoU(args.groundingdino_config_path, args.groundingdino_checkpoint_path, device)
    model = AutoModel.from_pretrained(
        args.vlm_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": device},
        torch_dtype=torch_dtype,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_id, trust_remote_code=True)

    score_instance_dir(
        img_dir=args.imgdir,
        instance_iou=instance_iou,
        model=model,
        tokenizer=tokenizer,
        iou_thr=args.iou_thr,
        save_detailed=args.save_detailed,
    )


if __name__ == "__main__":
    main()
