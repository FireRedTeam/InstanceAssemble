import argparse
from pathlib import Path
import torch
from src.utils import read_layout_json
from src.layout import Layout
from src.model_utils import load_instanceassemble_flux, load_instanceassemble_sd3


def parse_args():
    p = argparse.ArgumentParser("InstanceAssemble Inference (Flux / SD3)")
    p.add_argument("--model_type", type=str, required=True, choices=["fluxdev", "fluxschnell", "sd3"],
                   help="Backbone type: fluxdev, fluxschnell or sd3")
    p.add_argument("--model_path", type=str, default="",
                   help="Backbone path or HuggingFace model id")
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Checkpoint directory containing pytorch_lora_weights.safetensors and layout.pth")
    p.add_argument("--input_json", type=str, required=True,
                   help="Layout JSON with 'caption' and 'annos'")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory; defaults to output/fluxdev, output/fluxschnell or output/sd3 depending on model type")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed; defaults to 42")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--layout_scale", type=float, default=1.0)
    p.add_argument("--grounding_ratio", type=float, default=0.3,
                   help="Apply layout control for the first ratio of steps, e.g. 0.3 = 30%")
    p.add_argument("--max_objs", type=int, default=50,
                   help="Maximum number of objects in Layout")
    return p.parse_args()


def load_backend(model_type: str, model_path: str, ckpt_path: Path, dtype, device):
    """Load backend according to model_type and return components."""
    if model_type == "fluxdev":
        from diffusers.pipelines import FluxPipeline  # noqa: F401
        from src.flux.generate import generate as generate_fn
        from src.flux.transformer import LayoutFluxTransformer2DModel  # noqa: F401
        
        if model_path == "":
            model_path = "black-forest-labs/FLUX.1-dev"
        if ckpt_path == "":
            ckpt_path = "./pretrained/flux"
        pipe, layout_transformer = load_instanceassemble_flux(model_path, ckpt_path, dtype, device)
        return pipe, layout_transformer, generate_fn, 42, "output/fluxdev", 28

    elif model_type == "fluxschnell":
        from diffusers.pipelines import FluxPipeline  # noqa: F401
        from src.flux.generate import generate as generate_fn
        from src.flux.transformer import LayoutFluxTransformer2DModel  # noqa: F401
        
        if model_path == "":
            model_path = "black-forest-labs/FLUX.1-schnell"
        if ckpt_path == "":
            ckpt_path = "./pretrained/flux"
        pipe, layout_transformer = load_instanceassemble_flux(model_path, ckpt_path, dtype, device)
        return pipe, layout_transformer, generate_fn, 42, "output/fluxschnell", 4

    elif model_type == "sd3":
        from diffusers.pipelines import StableDiffusion3Pipeline  # noqa: F401
        from src.sd3.generate import generate as generate_fn
        from src.sd3.transformer import LayoutSD3Transformer2DModel  # noqa: F401

        if model_path == "":
            model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
        if ckpt_path == "":
            ckpt_path = "./pretrained/sd3"
        pipe, layout_transformer = load_instanceassemble_sd3(model_path, ckpt_path, dtype, device)
        return pipe, layout_transformer, generate_fn, 42, "output/sd3", 50

    else:
        raise ValueError(f"Model type '{model_type}' not supported")


@torch.inference_mode()
def run_inference(generate_fn, pipe, layout_transformer, prompt, layout_obj,
                  steps, layout_scale, grounding_ratio, H, W, seed, device):
    images = generate_fn(
        pipeline=pipe,
        layout_transformer=layout_transformer,
        prompt=prompt,
        generator=torch.Generator(device=device.type).manual_seed(seed),
        num_inference_steps=steps,
        layout_scale=layout_scale,
        grounding_ratio=grounding_ratio,
        layout=layout_obj,
        height=H,
        width=W,
    )
    return images.images[0]


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    pipe, layout_transformer, generate_fn, default_seed, default_outdir, default_steps = load_backend(
        args.model_type, args.model_path, args.ckpt_path, dtype, device
    )

    seed = args.seed if args.seed is not None else default_seed
    steps = args.steps if args.steps is not None else default_steps
    outdir_root = Path(args.outdir) if args.outdir is not None else Path(default_outdir)

    prompt, anno_feed, H, W, input_seed = read_layout_json(Path(args.input_json))
    seed = input_seed if input_seed is not None else seed
    layout_obj = Layout(anno_feed, max_objs=args.max_objs)

    image = run_inference(
        generate_fn, pipe, layout_transformer, prompt, layout_obj,
        steps=steps, layout_scale=args.layout_scale,
        grounding_ratio=args.grounding_ratio, H=H, W=W,
        seed=seed, device=device
    )

    save_name = Path(args.input_json).stem
    outdir = outdir_root / save_name
    outdir.mkdir(parents=True, exist_ok=True)
    image_path = outdir / f"{save_name}.jpg"
    layout_path = outdir / f"{save_name}_layout.jpg"
    image.save(image_path)
    layout_obj.show_layout_on_image(image).save(layout_path)

    print(f"[Success] saved: {image_path}")
    print(f"[Success] saved: {layout_path}")


if __name__ == "__main__":
    main()
