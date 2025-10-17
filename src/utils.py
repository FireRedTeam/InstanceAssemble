from pathlib import Path
import json
import colorsys
from typing import Dict, List, Tuple, Optional

def xyxy2xywh(bbox: List[int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2-x1, y2-y1

def read_layout_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    global_caption = data["prompt"]
    annos = data["annos"]

    anno_feed = []
    for cond in annos:
        anno_feed.append({
            "text": cond.get("caption", ""),
            "category": cond.get("category_name", ""),
            "bbox": xyxy2xywh(cond["bbox"]),
            "hw": [data["height"], data["width"]],
        })

    return global_caption, anno_feed, data["height"], data["width"], data.get("seed", None)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def generate_distinct_palette(max_objs: int) -> Tuple[List[str], List[Tuple[int,int,int]]]:
    if max_objs <= 0:
        return [], []

    tiers = [
        (0.90, 0.72),
        (0.65, 0.58),
        (0.98, 0.85),
    ]

    golden = 0.61803398875
    h0 = 0.19

    hex_list: List[str] = []
    bgr_list: List[Tuple[int,int,int]] = []

    order = []
    for offset in range(3):
        order.extend(range(offset, max_objs, 3))

    hues = [(h0 + k * golden) % 1.0 for k in range(max_objs)]

    for pos, k in enumerate(order):
        tier_idx = pos % 3
        s, v = tiers[tier_idx]
        h = hues[k]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        hex_list.append(_rgb_to_hex(r, g, b))
        bgr_list.append((int(b * 255), int(g * 255), int(r * 255)))

    return hex_list, bgr_list
