import sys, os, glob, json, argparse
sys.path.append("..")

import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from src.utils import xyxy2xywh
from src.model_utils import load_instanceassemble_sd3, load_instanceassemble_flux
from src.layout import Layout
from src.flux.generate import generate as generate_flux
from src.sd3.generate import generate as generate_sd3

dataset_repo = "FireRedTeam/DenseLayout"
MAX_OBJS = 100

def parse_args():
    p = argparse.ArgumentParser("InstanceAssemble DenseLayout Benchmark Generation")
    p.add_argument("--model_type", required=True, choices=["fluxdev", "fluxschnell", "sd3"])
    p.add_argument("--model_path", default="", help="Backbone path or HF id")
    p.add_argument("--ckpt_path", default="", help="Checkpoint dir with lora + layout.pth")
    p.add_argument("--outdir", required=True, help="Save dir, e.g. ./output/fluxdev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grounding_ratio", type=float, default=0.3,
                   help="Ratio of steps applying layout control, e.g. 0.3 = 30%")
    p.add_argument("--show_layout", action="store_true", help="If set, also save images with layout overlay")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    if args.model_type == "fluxdev":
        if args.model_path == "":
            args.model_path = "black-forest-labs/FLUX.1-dev"
        if args.ckpt_path == "":
            args.ckpt_path = "../pretrained/flux"
        args.steps = 28
        pipe, layout_transformer = load_instanceassemble_flux(args.model_path, args.ckpt_path, dtype, device)
        generate_fn = generate_flux
    elif args.model_type == "fluxschnell":
        if args.model_path == "":
            args.model_path = "black-forest-labs/FLUX.1-schnell"
        if args.ckpt_path == "":
            args.ckpt_path = "../pretrained/flux"
        args.steps = 4
        pipe, layout_transformer = load_instanceassemble_flux(args.model_path, args.ckpt_path, dtype, device)
        generate_fn = generate_flux
    elif args.model_type == "sd3":
        if args.model_path == "":
            args.model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
        if args.ckpt_path == "":
            args.ckpt_path = "../pretrained/sd3"
        args.steps = 50
        pipe, layout_transformer = load_instanceassemble_sd3(args.model_path, args.ckpt_path, dtype, device)
        generate_fn = generate_sd3
    else:
        raise ValueError(f"Model type '{args.model_type}' not supported")

    test_dataset = load_dataset(dataset_repo, split="test")

    out_img = os.path.join(args.outdir, "images_denselayout")
    out_img_with_layout = os.path.join(args.outdir, "images_denselayout_with_layout")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_img_with_layout, exist_ok=True)

    for data in tqdm(test_dataset):
        fname = f"{data['id']}.jpg"

        prompt = data["prompt"]

        anno_feed = [
            {
                "text": c.get("caption", ""),
                "category": c.get("category_name", ""),
                "bbox": xyxy2xywh(c["bbox"]),
                "hw": [data["height"], data["width"]],
            }
            for c in data["annos"][:MAX_OBJS]
        ]
        H, W = data["height"], data["width"]
        layout = Layout(anno_feed, max_objs=MAX_OBJS)

        with torch.no_grad():
            result = generate_fn(
                pipeline=pipe,
                layout_transformer=layout_transformer,
                prompt=[prompt],
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                num_inference_steps=args.steps,
                layout=layout,
                height=H,
                width=W,
                grounding_ratio=args.grounding_ratio,
            )
        img = result.images[0]

        img.save(os.path.join(out_img, fname))
        if args.show_layout:
            layout.show_layout_on_image(img).save(os.path.join(out_img_with_layout, fname))

    print(f"Results saved to {args.outdir}")
