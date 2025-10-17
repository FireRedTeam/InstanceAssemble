from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from safetensors.torch import load_file

from diffusers.pipelines import StableDiffusion3Pipeline, FluxPipeline
from .sd3.transformer import LayoutSD3Transformer2DModel
from .flux.transformer import LayoutFluxTransformer2DModel

def _zero_out_lora_scales(module: torch.nn.Module) -> None:
    """Set all active LoRA adapter scales to 0.0 (freeze LoRA influence)."""
    for _, m in module.named_modules():
        if hasattr(m, "set_scale"):
            adapters = getattr(m, "active_adapters", None)
            if adapters:
                for adp in list(adapters):
                    m.set_scale(adp, 0.0)


def _require_file(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")

def load_instanceassemble_sd3(
    model_path: str,
    ckpt_dir: str | Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[StableDiffusion3Pipeline, torch.nn.Module]:
    """
    Return (StableDiffusion3Pipeline, LayoutSD3Transformer2DModel) with LoRA+layout weights loaded.
    """
    # backbone
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)

    # layout transformer (init from backbone transformer)
    layout_transformer = LayoutSD3Transformer2DModel.from_config(pipe.transformer.config, attention_type="layout")
    layout_transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)

    # weights
    ckpt_dir = Path(ckpt_dir)
    lora_file = ckpt_dir / "pytorch_lora_weights.safetensors"
    layout_file = ckpt_dir / "layout.pth"
    _require_file(lora_file, "LoRA weights")
    _require_file(layout_file, "layout state dict")

    lora_sd = load_file(lora_file)
    StableDiffusion3Pipeline.load_lora_into_transformer(state_dict=lora_sd, transformer=layout_transformer)

    layout_sd = torch.load(layout_file, map_location="cpu")
    layout_transformer.load_state_dict(layout_sd, strict=False)

    _zero_out_lora_scales(layout_transformer)
    layout_transformer = layout_transformer.to(dtype=dtype, device=device)
    return pipe, layout_transformer

def load_instanceassemble_flux(
    model_path: str,
    ckpt_dir: str | Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[FluxPipeline, torch.nn.Module]:
    """
    Return (FluxPipeline, LayoutFluxTransformer2DModel) with LoRA(alphas)+layout weights loaded.
    """
    # backbone
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)

    # layout transformer (init from backbone transformer)
    layout_transformer = LayoutFluxTransformer2DModel.from_config(pipe.transformer.config, attention_type="layout")
    layout_transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)

    # weights
    ckpt_dir = Path(ckpt_dir)
    lora_file = ckpt_dir / "pytorch_lora_weights.safetensors"
    layout_file = ckpt_dir / "layout.pth"
    _require_file(lora_file, "LoRA weights")
    _require_file(layout_file, "layout state dict")

    lora_sd, alphas = FluxPipeline.lora_state_dict(
        pretrained_model_name_or_path_or_dict=str(ckpt_dir),
        weight_name=lora_file.name,
        return_alphas=True,
    )
    FluxPipeline.load_lora_into_transformer(
        state_dict=lora_sd,
        network_alphas=alphas,
        transformer=layout_transformer,
    )

    layout_sd = torch.load(layout_file, map_location="cpu")
    layout_transformer.load_state_dict(layout_sd, strict=False)

    _zero_out_lora_scales(layout_transformer)
    layout_transformer = layout_transformer.to(dtype=dtype, device=device)
    return pipe, layout_transformer