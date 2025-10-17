import math
import colorsys
import cv2
import numpy as np
import torch.nn as nn
import torch
from PIL import Image

from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.pipelines import StableDiffusion3Pipeline, FluxPipeline

from .utils import generate_distinct_palette

class Layout:
    def __init__(self, conds, max_objs: int = 50, point_num: int = 36):
        self.hw = None
        self.max_objs = max_objs
        self.point_num = point_num

        self.cond_masks = torch.zeros(max_objs)  # binary mask: whether instance i exists
        self.categorys = [''] * max_objs
        self.texts = [''] * max_objs
        self.boxes = torch.zeros(max_objs, point_num * 2)

        self.set_conds(conds)
    
    def set_conds(self, annos):
        for i, anno in enumerate(annos):
            if i >= self.max_objs:
                break
            self.cond_masks[i] = 1
            self.categorys[i] = anno['category']
            self.texts[i] = f"{anno['category']}"
            if anno.get('text'):
                self.texts[i] += f", {anno['text']}"

            bbox = anno['bbox']
            hw = anno["hw"]
            # (x1, y1, x2, y2) normalized to [0,1]
            self.boxes[i][0] = (bbox[0]) / hw[1]
            self.boxes[i][1] = (bbox[1]) / hw[0]
            self.boxes[i][2] = (bbox[0] + bbox[2]) / hw[1]
            self.boxes[i][3] = (bbox[1] + bbox[3]) / hw[0]

        self.dense_sample()

    def dense_sample(self):
        point_per_line = int(math.sqrt(self.point_num))
        for i in range(self.max_objs):
            if self.cond_masks[i] == 0:
                break
            x1, y1, x2, y2 = self.boxes[i][:4]
            step_x = (x2 - x1) / (point_per_line - 1)
            step_y = (y2 - y1) / (point_per_line - 1)
            for u in range(point_per_line):
                for v in range(point_per_line):
                    now_idx = (u * point_per_line + v) * 2
                    self.boxes[i][now_idx] = x1 + u * step_x
                    self.boxes[i][now_idx + 1] = y1 + v * step_y

    def show_layout_on_image(self, image: Image.Image) -> Image.Image:
        """Draw boxes and category labels on PIL image."""
        def draw(img_bgr: np.ndarray, boxes: torch.Tensor, labels: list[str]) -> np.ndarray:
            h, w, _ = img_bgr.shape
            _, bgr = generate_distinct_palette(self.max_objs)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_color = (255, 255, 255)

            for i in range(len(boxes)):
                color = bgr[i % len(bgr)] if bgr else (0, 255, 0)
                (x1, y1), (x2, y2) = boxes[i][:2], boxes[i][-2:]
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

                label = labels[i]
                sz, _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(img_bgr, (x1, y1 - sz[1] - 10), (x1 + sz[0], y1), color, -1)
                cv2.putText(img_bgr, label, (x1, y1 - 7), font, 0.5, label_color, 1)

            return img_bgr

        if isinstance(image, Image.Image):
            arr = np.array(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            bgr = draw(bgr, self.boxes.clone(), self.categorys)
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        raise NotImplementedError("Only PIL.Image is supported here.")

def bbox_to_mask(bbox: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    mask = torch.zeros((latent_h, latent_w), device=bbox.device)
    (x1, y1), (x2, y2) = bbox[:2], bbox[-2:]
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return mask
    x1, y1, x2, y2 = int(x1 * latent_w), int(y1 * latent_h), int(x2 * latent_w), int(y2 * latent_h)
    if x1 == x2:
        x2 = x1 + 1
    if y1 == y2:
        y2 = y1 + 1
    mask[y1:y2, x1:x2] = 1
    return mask

def get_layout_idxslist(layout_boxes: torch.Tensor, latent_hw: tuple[int, int]):
    latent_h, latent_w = latent_hw
    img_idxs_list = []
    for obj_i in range(layout_boxes.shape[0]):
        mask_obj = bbox_to_mask(layout_boxes[obj_i], latent_h, latent_w)
        img_idxs = mask_obj.view(-1).nonzero().to(torch.int).view(-1)
        img_idxs_list.append(img_idxs)
    return img_idxs_list


def get_text_ids(layout_boxes: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    max_objs = layout_boxes.shape[0]
    text_ids = torch.zeros((max_objs, 3))
    for obj_i in range(max_objs):
        (x1, y1), (x2, y2) = layout_boxes[obj_i][:2], layout_boxes[obj_i][-2:]
        x1, y1, x2, y2 = int(x1 * latent_w), int(y1 * latent_h), int(x2 * latent_w), int(y2 * latent_h)
        text_ids[obj_i, 1] = (y1 + y2) / 2  # height
        text_ids[obj_i, 2] = (x1 + x2) / 2  # width
    return text_ids


def encode_layout_flux(layout: Layout, pipe: FluxPipeline, latent_hw: tuple[int, int]):
    prev_level = logger.level
    try:
        logger.setLevel(logging.ERROR)
        if layout.cond_masks.sum() == 0:
            return None

        device = pipe.device
        with torch.no_grad():
            text_embeddings = pipe._get_clip_prompt_embeds(
                prompt=layout.texts, device=device, num_images_per_prompt=1
            )

        text_ids = get_text_ids(layout.boxes, latent_hw[0], latent_hw[1])
        img_idxs_list = get_layout_idxslist(layout.boxes, latent_hw)

        boxes = layout.boxes.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        text_embeddings = text_embeddings.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        text_ids = text_ids.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        masks = layout.cond_masks.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)

        layout_kwargs = {
            "layout": {
                "boxes": boxes,
                "text_embeddings": text_embeddings,
                "text_ids": text_ids,
                "masks": masks,
                "img_idxs_list": img_idxs_list,
            }
        }
        return layout_kwargs
    finally:
        logger.setLevel(prev_level)

def encode_layout_sd3(layout:Layout, pipe: StableDiffusion3Pipeline, latent_hw: tuple[int, int]):
    if layout.cond_masks.sum() == 0:
        return None
    device = pipe.device
    max_objs = layout.max_objs
    n_objs = int(layout.cond_masks.sum().item())

    tokenizer_inputs = pipe.tokenizer(
        layout.texts,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_embeddings_1 = pipe.text_encoder(tokenizer_inputs, output_hidden_states=True)[0]
    tokenizer_inputs_2 = pipe.tokenizer_2(
        layout.texts,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_embeddings_2 = pipe.text_encoder_2(tokenizer_inputs_2, output_hidden_states=True)[0]
    text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1) 

    boxes = layout.boxes.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
    text_embeddings = text_embeddings.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
    masks = layout.cond_masks.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
    img_idxs_list = [get_layout_idxslist(layout.boxes, latent_hw)]
    if pipe.do_classifier_free_guidance:
        boxes = torch.cat([boxes] * 2)
        text_embeddings = torch.cat([text_embeddings] * 2)
        masks = torch.cat([masks] * 2)
        masks[: 1] = 0
        img_idxs_list = [[]] + img_idxs_list

    layout_kwargs = {
            "layout": {"boxes": boxes, "text_embeddings": text_embeddings, "masks": masks, "img_idxs_list": img_idxs_list}
        } 

    return layout_kwargs

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_fourier_embeds_from_boundingbox(embed_dim, box, position_dim):
    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, position_dim)

    return emb

class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
        
class TextBoundingboxProjection(nn.Module):
    def __init__(self, pooled_projection_dim, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 36*2  # 2: sin/cos, 36*2: xyxy 

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, positive_len, act_fn="silu")
        self.linears = PixArtAlphaTextProjection(in_features=self.positive_len + self.position_dim,hidden_size=out_dim//2,out_features=out_dim, act_fn="silu")

    def forward(
        self,
        boxes,
        masks,
        positive_embeddings, 
    ):
        
        masks = masks.unsqueeze(-1) 

        # embedding position
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes, self.position_dim) 

        xyxy_embedding = xyxy_embedding * masks

        positive_embeddings = self.text_embedder(positive_embeddings) 

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks

        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))

        return objs
