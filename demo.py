import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from PIL import Image, ImageDraw
import colorsys
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from src.layout import Layout
from src.flux.transformer import LayoutFluxTransformer2DModel
from src.flux.generate import generate as flux_generate
from src.utils import generate_distinct_palette, xyxy2xywh
from diffusers.pipelines import FluxPipeline

from src.sd3.transformer import LayoutSD3Transformer2DModel
from src.sd3.generate import generate as sd3_generate
from diffusers.pipelines import StableDiffusion3Pipeline
from safetensors.torch import load_file

# ---------------------- Config ----------------------
st.set_page_config(page_title="InstanceAssemble Demo (textual-only)", layout="wide")

MAX_OBJS = 50
default_seed = 42
default_height = 1024
default_width = 1024
PALETTE, bgr = generate_distinct_palette(MAX_OBJS)

# Keys for st.session_state
SS_BOXES = "boxes"                 
SS_ADDED_SET = "added_bbox_set"    
SS_CANVAS_KEY = "canvas_key"       
SS_PROMPT = "global_prompt"
SS_SEED = "seed"
SS_HEIGHT = "height"
SS_WIDTH = "width"
SS_PENDING_EXAMPLE = "pending_example"  
SS_PIPE = "loaded_pipe"
SS_LT = "loaded_layout_transformer"
SS_LOADED = "is_model_loaded"
SS_LOADED_CFG = "loaded_cfg_triple" 

# ---------------------- Preset Examples (for bottom Examples column) ----------------------
def load_presets_from_demo(demo_dir: str = "./demo"):
    presets = []
    demo_path = Path(demo_dir)

    for json_file in demo_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        boxes = []
        for box in data.get("annos", []):
            boxes.append({
                "bbox": box["bbox"],
                "category": box.get("category_name", ""),
                "caption": box.get("caption", "")
            })

        preset = {
            "name": json_file.stem,
            "height": data.get("height", default_height),
            "width": data.get("width", default_width),
            "prompt": data.get("prompt", ""),
            "annos": boxes,
            "seed": data.get("seed", default_seed)
        }
        presets.append(preset)

    return presets

PRESET_CASES = load_presets_from_demo("./demo")


@dataclass
class GenSettings:
    model_type: str
    model_path: str
    ckpt_path: str
    gen_H: int
    gen_W: int
    steps: int
    layout_scale: float
    grounding_ratio: float
    seed: int
    canvas_w: int
    stroke_width: int
    device: torch.device
    dtype: torch.dtype = torch.bfloat16

def _rgba(hex_rgb: str, alpha: float) -> str:
    if hex_rgb.startswith("#") and len(hex_rgb) == 7:
        return f"{hex_rgb}{int(alpha * 255):02x}"
    return hex_rgb

def _hex_to_rgba_str(hex_rgb: str, a: float = 0.12) -> str:
    """#RRGGBB -> 'rgba(r,g,b,a)'"""
    h = hex_rgb.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

def _build_initial_fabric_from_boxes(
    boxes: List[Dict], view_W: int, view_H: int, gen_W: int, gen_H: int, stroke_w: int
) -> Dict:
    if gen_W <= 0 or gen_H <= 0 or view_W <= 0 or view_H <= 0:
        return {"objects": []}

    sx = view_W / float(gen_W)
    sy = view_H / float(gen_H)

    objects = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        left  = float(x1) * sx
        top   = float(y1) * sy
        width = float(x2 - x1) * sx
        height= float(y2 - y1) * sy
        color = b.get("color", "#000000")
        objects.append({
            "type": "rect",
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "fill": _hex_to_rgba_str(color, 0.12),
            "stroke": color,
            "strokeWidth": float(stroke_w),
            "selectable": False,
            "evented": False,
        })
    return {"objects": objects}

def scale_rect_to_gen_space(o: dict, sx: float, sy: float) -> Tuple[int, int, int, int]:
    """Scale a rect (fabric.js object) from canvas coords into generation-space coords."""
    left = float(o.get("left", 0))
    top = float(o.get("top", 0))
    w = float(o.get("width", 0)) * float(o.get("scaleX", 1))
    h = float(o.get("height", 0)) * float(o.get("scaleY", 1))
    x1, y1 = int(round(left * sx)), int(round(top * sy))
    x2, y2 = int(round((left + w) * sx)), int(round((top + h) * sy))
    return x1, y1, x2, y2

def ensure_session_state():
    if SS_BOXES not in st.session_state:
        st.session_state[SS_BOXES] = []
    if SS_ADDED_SET not in st.session_state:
        st.session_state[SS_ADDED_SET] = set()
    if SS_CANVAS_KEY not in st.session_state:
        st.session_state[SS_CANVAS_KEY] = 0
    if SS_PROMPT not in st.session_state:
        st.session_state[SS_PROMPT] = ""
    if SS_SEED not in st.session_state:
        st.session_state[SS_SEED] = default_seed
    if SS_HEIGHT not in st.session_state:
        st.session_state[SS_HEIGHT] = default_height
    if SS_WIDTH not in st.session_state:
        st.session_state[SS_WIDTH] = default_width
    if SS_PENDING_EXAMPLE not in st.session_state:
        st.session_state[SS_PENDING_EXAMPLE] = None
    if SS_PIPE not in st.session_state:
        st.session_state[SS_PIPE] = None
    if SS_LT not in st.session_state:
        st.session_state[SS_LT] = None
    if SS_LOADED not in st.session_state:
        st.session_state[SS_LOADED] = False
    if SS_LOADED_CFG not in st.session_state:
        st.session_state[SS_LOADED_CFG] = None


def next_color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]

def _reset_on_model_type_change():
    mt = st.session_state["model_type"]
    if mt == "Flux.1-dev":
        st.session_state["steps"] = 28
        st.session_state["ckpt_path"] = "./pretrained/flux"
    elif mt == "Flux.1-schnell":
        st.session_state["steps"] = 4
        st.session_state["ckpt_path"] = "./pretrained/flux"
    else:  # SD3
        st.session_state["steps"] = 50
        st.session_state["ckpt_path"] = "./pretrained/sd3"


def _zero_out_lora_scales(module: torch.nn.Module) -> None:
    for _, m in module.named_modules():
        if hasattr(m, "set_scale"):
            adapters = getattr(m, "active_adapters", None)
            if adapters:
                for adp in list(adapters):
                    m.set_scale(adp, 0.0)

def _build_flux_layout_transformer_from_ckpt(pipe, ckpt_dir: Path, dtype: torch.dtype, device: torch.device):
    layout_transformer = LayoutFluxTransformer2DModel.from_config(pipe.transformer.config, attention_type="layout")
    layout_transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)

    lora_file = ckpt_dir / "pytorch_lora_weights.safetensors"
    if not lora_file.exists():
        raise FileNotFoundError(f"Missing LoRA: {lora_file}")
    lora_sd, alphas = FluxPipeline.lora_state_dict(
        pretrained_model_name_or_path_or_dict=str(ckpt_dir),
        weight_name=lora_file.name,
        return_alphas=True,
    )
    FluxPipeline.load_lora_into_transformer(state_dict=lora_sd, network_alphas=alphas, transformer=layout_transformer)

    layout_file = ckpt_dir / "layout.pth"
    if not layout_file.exists():
        raise FileNotFoundError(f"Missing layout state dict: {layout_file}")
    layout_transformer.load_state_dict(torch.load(layout_file, map_location="cpu"), strict=False)

    _zero_out_lora_scales(layout_transformer)
    return layout_transformer.to(dtype=dtype, device=device)


def _build_sd3_layout_transformer_from_ckpt(pipe, ckpt_dir: Path, dtype: torch.dtype, device: torch.device):
    layout_transformer = LayoutSD3Transformer2DModel.from_config(pipe.transformer.config, attention_type="layout")
    layout_transformer.load_state_dict(pipe.transformer.state_dict(), strict=False)

    lora_file = ckpt_dir / "pytorch_lora_weights.safetensors"
    if not lora_file.exists():
        raise FileNotFoundError(f"Missing LoRA: {lora_file}")
    lora_sd = load_file(lora_file)
    StableDiffusion3Pipeline.load_lora_into_transformer(state_dict=lora_sd, transformer=layout_transformer)

    layout_file = ckpt_dir / "layout.pth"
    if not layout_file.exists():
        raise FileNotFoundError(f"Missing layout state dict: {layout_file}")
    layout_transformer.load_state_dict(torch.load(layout_file, map_location="cpu"), strict=False)

    _zero_out_lora_scales(layout_transformer)
    return layout_transformer.to(dtype=dtype, device=device)


@st.cache_resource(show_spinner=False)
def load_flux_bundle(model_path: str, ckpt_path: str, dtype: torch.dtype, device: torch.device):
    """Load Flux backbone and (if ckpt_path valid) layout transformer with weights."""
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.transformer.to('cpu')
    ckpt_dir = Path(ckpt_path) if ckpt_path else None
    if ckpt_dir and ckpt_dir.exists():
        layout_transformer = _build_flux_layout_transformer_from_ckpt(pipe, ckpt_dir, dtype, device)
    return pipe, layout_transformer


@st.cache_resource(show_spinner=False)
def load_sd3_bundle(model_path: str, ckpt_path: str, dtype: torch.dtype, device: torch.device):
    """Load SD3 backbone and (if ckpt_path valid) layout transformer with weights."""
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.transformer.to('cpu')
    ckpt_dir = Path(ckpt_path) if ckpt_path else None
    if ckpt_dir and ckpt_dir.exists():
        layout_transformer = _build_sd3_layout_transformer_from_ckpt(pipe, ckpt_dir, dtype, device)
    return pipe, layout_transformer


def sidebar_controls() -> GenSettings:
    with st.sidebar:
        st.subheader("Model Setting")
        model_type = st.selectbox("Backbone Model", ["Flux.1-dev", "Flux.1-schnell", "SD3"], index=0, key="model_type", on_change=_reset_on_model_type_change)
        _reset_on_model_type_change()
        model_path = st.text_input("model_id/model_path（Backbone）", value="black-forest-labs/FLUX.1-dev")
        ckpt_path = st.text_input("ckpt_path", value=st.session_state["ckpt_path"], help="including pytorch_lora_weights.safetensors and layout.pth")


        cfg_triple = (model_type, model_path.strip(), ckpt_path.strip())

        if st.session_state[SS_LOADED] and st.session_state[SS_LOADED_CFG] != cfg_triple:
            st.info("model settings changed: please click Load model below to reload.")

        if st.button("📦 Load model", use_container_width=True):
            try:
                if st.session_state[SS_PIPE] is not None:
                    st.session_state[SS_PIPE].to('cpu')
                    del st.session_state[SS_PIPE]
                if st.session_state[SS_LT] is not None:
                    st.session_state[SS_LT].to('cpu')
                    del st.session_state[SS_LT]
                torch.cuda.empty_cache()
                
                st.session_state[SS_PIPE] = None
                st.session_state[SS_LT] = None
                st.session_state[SS_LOADED] = False

                _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                _dtype = torch.bfloat16

                if model_type == "Flux.1-dev" or model_type == "Flux.1-schnell":
                    pipe, layout_transformer = load_flux_bundle(model_path.strip(), ckpt_path.strip(), _dtype, _device)
                else:
                    pipe, layout_transformer = load_sd3_bundle(model_path.strip(), ckpt_path.strip(), _dtype, _device)

                st.session_state[SS_PIPE] = pipe
                st.session_state[SS_LT] = layout_transformer
                st.session_state[SS_LOADED_CFG] = cfg_triple
                st.session_state[SS_LOADED] = True
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Model loading failed.")

        st.subheader("Inference Setting")
        col_hw1, col_hw2 = st.columns(2)
        with col_hw1:
            gen_H = st.number_input("Generate Height", min_value=256, max_value=3096, value=st.session_state.get(SS_HEIGHT, default_height), step=64, on_change=st.rerun)
        with col_hw2:
            gen_W = st.number_input("Generate Width", min_value=256, max_value=3096, value=st.session_state.get(SS_WIDTH, default_width), step=64, on_change=st.rerun)

        steps = st.slider("num_inference_steps", 5, 100, st.session_state["steps"], 1)
        layout_scale = st.slider("layout_scale", 0.0, 3.0, 1.0, 0.05)
        grounding_ratio = st.slider("grounding_ratio", 0.0, 1.0, 0.3, 0.05, help="Apply layout control during the first X%% of the steps")
        seed = st.number_input("seed", value=st.session_state[SS_SEED], step=1)

        st.markdown("---")
        st.subheader("Canvas Setting")
        canvas_w = st.slider(
            "Canvas display width (px)",
            min_value=120, max_value=1000, value=400, step=10,
            help="Only affects the visualization size; the coordinates will be proportionally mapped to the generation resolution.",
            on_change=st.rerun
        )
        stroke_width = st.slider("Stroke width", 1, 6, 2, 1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.caption(f"Device: **{device}**；dtype: bfloat16")

    return GenSettings(
        model_type=model_type,
        model_path=model_path,
        ckpt_path=ckpt_path,
        gen_H=int(gen_H),
        gen_W=int(gen_W),
        steps=int(steps),
        layout_scale=float(layout_scale),
        grounding_ratio=float(grounding_ratio),
        seed=int(seed),
        canvas_w=int(canvas_w),
        stroke_width=int(stroke_width),
        device=device,
    )


def panel_canvas_and_table(settings: GenSettings) -> str:
    """Left column: canvas + Add Instance + editable table."""
    ensure_session_state()

    st.title("🖼️ Canvas")
    st.markdown(
        "<p style='font-size:16px; color:gray;'>Draw a box first ➜ click Add Instance ➜ enter the instance category and caption</p>",
        unsafe_allow_html=True
    )

    st.text_area(
        "Global Prompt",
        key=SS_PROMPT,
        height=88,
    )
    prompt = st.session_state[SS_PROMPT]

    # Visual canvas size (purely UI), then map to gen size.
    view_W = int(settings.canvas_w)
    _ratio = (view_W / settings.gen_W) if int(settings.gen_W) > 0 else 1.0
    view_H = int(round(settings.gen_H * _ratio))

    next_color = next_color_for(len(st.session_state[SS_BOXES]))


    if st.button("🧹 Clear canvas and added instances", type="secondary"):
        st.session_state[SS_CANVAS_KEY] += 1
        st.session_state[SS_BOXES].clear()
        st.session_state[SS_ADDED_SET].clear()
        st.success("Cleared")
        st.rerun()

    initial_fabric = _build_initial_fabric_from_boxes(
        st.session_state[SS_BOXES],
        view_W=view_W,
        view_H=view_H,
        gen_W=settings.gen_W,
        gen_H=settings.gen_H,
        stroke_w=settings.stroke_width,
    )

    canvas = st_canvas(
        fill_color=_rgba(next_color, 0.12),
        stroke_color=next_color,
        background_color="#f7f7f9",
        update_streamlit=True,
        height=int(view_H),
        width=int(view_W),
        drawing_mode="rect",
        stroke_width=settings.stroke_width,
        key=f"canvas_{st.session_state[SS_CANVAS_KEY]}",
        initial_drawing=initial_fabric,
        display_toolbar=False,
    )

    # extract last rect
    raw_last_rect = None
    if canvas and canvas.json_data:
        rect_objs = [
            o for o in canvas.json_data.get("objects", [])
            if o.get("type") == "rect" and o.get("selectable", True)  # 只要可选（=用户新画的）
        ]
        if rect_objs:
            raw_last_rect = rect_objs[-1]

    sx, sy = settings.gen_W / float(view_W), settings.gen_H / float(view_H)

    if st.button("➕ Add Instance", type="primary", width='stretch'):
        if raw_last_rect is None:
            st.warning("Please draw a rectangle on the canvas first, then click Add Instance.")
        else:
            x1, y1, x2, y2 = scale_rect_to_gen_space(raw_last_rect, sx, sy)
            bbox_t = (x1, y1, x2, y2)
            if bbox_t in st.session_state[SS_ADDED_SET]:
                st.info("This rectangle has already been added (same coordinates), no need to add again.")
            else:
                color = next_color_for(len(st.session_state[SS_BOXES]))
                st.session_state[SS_BOXES].append({
                    "bbox": [x1, y1, x2, y2],
                    "category": "",
                    "caption": "",
                    "color": color,
                })
                st.session_state[SS_ADDED_SET].add(bbox_t)
                st.success(f"Added an instance: {bbox_t}")
                st.rerun()

    # Editable table
    st.subheader("✅ Added instance ")
    st.markdown(
        "<p style='font-size:16px; color:gray;'>(please provide at least one field: category or caption, then press Save Layout)</p>",
        unsafe_allow_html=True
    )
    if len(st.session_state[SS_BOXES]) == 0:
        st.info("No Instance. Pipeline: Draw a box first ➜ click Add Instance ➜ enter the instance category and caption")
    else:
        rows = []
        for i, a in enumerate(st.session_state[SS_BOXES]):
            x1, y1, x2, y2 = a["bbox"]
            rows.append({
                "id": i + 1,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "category": a.get("category", ""),
                "caption": a.get("caption", ""),
                "color": a.get("color", next_color_for(i)),
            })
        df = pd.DataFrame(rows, columns=["id", "x1", "y1", "x2", "y2", "category", "caption", "color"])

        gob = GridOptionsBuilder.from_dataframe(df)
        gob.configure_default_column(editable=False, resizable=True, sortable=False, filter=False)
        for col in ["id", "x1", "y1", "x2", "y2", "color"]:
            gob.configure_column(col, editable=False, width=50)
        gob.configure_column("category", header_name="category", editable=True, width=120, cellEditor="agTextCellEditor")
        gob.configure_column("caption", header_name="caption", editable=True, minwidth=120, cellEditor="agTextCellEditor")

        row_style_js = JsCode(
            """
function(params) {
  const c = params.data.color || '#000000';
  return { color: c };
}
"""
        )
        gob.configure_grid_options(getRowStyle=row_style_js)

        color_cell_style_js = JsCode(
            """
function(params) {
  const c = params.value || '#000000';
  function yiq(hex){
    const r=parseInt(hex.substr(1,2),16),
          g=parseInt(hex.substr(3,2),16),
          b=parseInt(hex.substr(5,2),16);
    const y=(r*299+g*587+b*114)/1000;
    return y >= 128 ? '#000000' : '#ffffff';
  }
  return { backgroundColor: c, color: yiq(c), textAlign: 'center', fontWeight: '600' };
}
"""
        )
        gob.configure_column("color", header_name="color", width=70, cellStyle=color_cell_style_js)

        gob.configure_grid_options(
            singleClickEdit=True,
            suppressClickEdit=False,
            stopEditingWhenCellsLoseFocus=True,
            enterMovesDown=True,
            enterMovesDownAfterEdit=True,
        )
        gob.configure_selection("none")
        gob.configure_grid_options(suppressDragLeaveHidesColumns=True)
        grid_options = gob.build()

        grid = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            height=300,
            theme="streamlit",
        )

        
    cols_btn = st.columns(2)
    with cols_btn[0]:
        if st.button("Save Layout", type="secondary", width="stretch"):
            edited_df = pd.DataFrame(grid["data"])  # sync back
            st.session_state[SS_BOXES] = [
                {
                    "bbox": [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])],
                    "category": (str(r["category"]) if pd.notna(r["category"]) else "").strip(),
                    "caption": (str(r["caption"]) if pd.notna(r["caption"]) else "").strip(),
                    "color": str(r["color"]) if pd.notna(r["color"]) else PALETTE[0],
                }
                for _, r in edited_df.iterrows()
            ]

    with cols_btn[1]:
        if len(st.session_state[SS_BOXES]) > 0:
            layout_json = {
                "caption": st.session_state[SS_PROMPT],
                "height": settings.gen_H,
                "width": settings.gen_W,
                "annos": [
                    {
                        "bbox": b["bbox"],
                        "caption": b.get("caption", ""),
                        "category_name": b.get("category", "")
                    }
                    for b in st.session_state[SS_BOXES]
                ],
                "seed": settings.seed
            }
            layout_str = json.dumps(layout_json, indent=2, ensure_ascii=False)
            st.download_button(
                "Download Layout JSON",
                data=layout_str,
                file_name="layout.json",
                mime="application/json",
                width='stretch'
            )

    return prompt


def validate_boxes(boxes: List[Dict]) -> List[int]:
    """Return 1-based row indices that are invalid (missing both caption and category)."""
    invalid_rows = []
    for i, b in enumerate(boxes):
        if (not b.get("caption")) and (not b.get("category")):
            invalid_rows.append(i + 1)
    return invalid_rows


def build_layout_from_boxes(boxes: List[Dict], H: int, W: int) -> Layout:
    anno_feed = [
        {
            "text": b["caption"],
            "category": b["category"],
            "bbox": xyxy2xywh(b["bbox"]),
            "hw": [int(H), int(W)],
        }
        for b in boxes
    ]
    return Layout(anno_feed, max_objs=MAX_OBJS)

def panel_inference_and_result(settings: GenSettings, prompt: str) -> None:
    st.title("⚙️ Inference")

    pipe = st.session_state.get(SS_PIPE)
    layout_transformer = st.session_state.get(SS_LT)
    is_ready = st.session_state.get(SS_LOADED, False)

    if not is_ready or pipe is None:
        st.info("Please first set the model type and path on the left, then click Load model.")
        return
    if layout_transformer is None:
        st.error("Failed to build/load (please check the ckpt path and reload from the left).")
        return

    # Validation
    can_run = (
        is_ready
        and (pipe is not None)
        and (layout_transformer is not None)
        and (len(st.session_state[SS_BOXES]) > 0)
        and (prompt != "")
    )
    if can_run:
        invalid_rows = validate_boxes(st.session_state[SS_BOXES])
        if invalid_rows:
            can_run = False
            st.error(f"Row {invalid_rows} has neither category nor caption filled in (at least one is required).")

    run = st.button("🚀 Generate", type="primary", width='stretch', disabled=not can_run)

    if not run:
        return

    try:
        layout_obj = build_layout_from_boxes(st.session_state[SS_BOXES], settings.gen_H, settings.gen_W)

        with st.spinner("Inferencing ..."):
            g = torch.Generator(device=settings.device.type).manual_seed(int(settings.seed))
            if settings.model_type == "Flux.1-dev" or settings.model_type == "Flux.1-schnell":
                res = flux_generate(
                    pipeline=pipe,
                    layout_transformer=layout_transformer,
                    prompt=prompt,
                    generator=g,
                    num_inference_steps=int(settings.steps),
                    layout_scale=float(settings.layout_scale),
                    grounding_ratio=float(settings.grounding_ratio),
                    layout=layout_obj,
                    height=int(settings.gen_H),
                    width=int(settings.gen_W),
                )
            else:
                res = sd3_generate(
                    pipeline=pipe,
                    layout_transformer=layout_transformer,
                    prompt=prompt,
                    generator=g,
                    num_inference_steps=int(settings.steps),
                    layout_scale=float(settings.layout_scale),
                    grounding_ratio=float(settings.grounding_ratio),
                    layout=layout_obj,
                    height=int(settings.gen_H),
                    width=int(settings.gen_W),
                )
            out_img: Optional[Image.Image] = res.images[0]

        if out_img is not None:
            st.image(out_img, caption="Result", width='stretch')
            st.image(
                layout_obj.show_layout_on_image(out_img),
                caption="Result with Layout",
                width='stretch',
            )

            ts = int(time.time())
            fn = f"gen_{settings.model_type.lower()}_{ts}.jpg"
            p = Path(f"/tmp/{fn}")
            out_img.save(p)
            st.download_button(
                "Download", data=open(p, "rb").read(), file_name=fn, mime="image/jpeg"
            )

    except Exception as e:
        st.exception(e)


# ---------------------- Pending Example: apply-on-start helpers ----------------------
def _apply_example_case_to_session(case: Dict):
    """Apply example data into session. MUST be called before any widget using these keys is created."""
    # prompt
    st.session_state[SS_PROMPT] = case.get("prompt", "")

    # boxes with colors
    boxes_raw = case.get("annos", [])
    new_boxes = []
    for i, b in enumerate(boxes_raw):
        col = next_color_for(i)
        new_boxes.append({
            "bbox": [int(x) for x in b["bbox"]],
            "category": str(b.get("category", "")),
            "caption": str(b.get("caption", "")),
            "color": col,
        })
    st.session_state[SS_BOXES] = new_boxes
    st.session_state[SS_ADDED_SET] = {tuple(b["bbox"]) for b in new_boxes}
    st.session_state[SS_SEED] = case.get("seed", st.session_state[SS_SEED])
    st.session_state[SS_HEIGHT] = case.get("height", st.session_state[SS_HEIGHT])
    st.session_state[SS_WIDTH] = case.get("width", st.session_state[SS_WIDTH])

    # bump canvas key to redraw with first color based on current len(boxes)
    st.session_state[SS_CANVAS_KEY] += 1


# ---------------------- Examples Panel (bottom) ----------------------
def _load_example_into_session(case: Dict):
    """Queue preset example into session (pending) then rerun.
    Actual write to SS_PROMPT/SS_BOXES happens at the beginning of main()
    BEFORE any widgets are instantiated.
    """
    ensure_session_state()
    st.session_state[SS_PENDING_EXAMPLE] = case   # <-- queue only
    st.rerun()                                    # <-- trigger rerun to apply early next run


def panel_examples() -> None:
    st.markdown("---")
    st.subheader("🧩 Examples")

    cols = st.columns(3)
    for i, case in enumerate(PRESET_CASES):
        with cols[i % 3]:
            if st.button(f"{case['name']}", key=f"example_{i}", width='stretch'):
                _load_example_into_session(case)


# ---------------------- App Entry ----------------------
def main() -> None:
    ensure_session_state()

    pending = st.session_state.get(SS_PENDING_EXAMPLE, None)
    if pending is not None:
        _apply_example_case_to_session(pending)
        st.session_state[SS_PENDING_EXAMPLE] = None
        st.toast(f"Loaded: {pending.get('name','')}", icon="✅")

    settings = sidebar_controls()
    left, right = st.columns([1.05, 1.0])
    with left:
        prompt = panel_canvas_and_table(settings)
    with right:
        panel_inference_and_result(settings, prompt)

    panel_examples()


if __name__ == "__main__":
    main()
