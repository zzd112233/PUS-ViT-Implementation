"""
Full-scene sliding-window inference for PUS-ViT Stage-2.

Requires Stage-1 (`pgcl_model_best_*.pth`) and Stage-2 (`fusion_model_best_*.pth`)
checkpoints. Paths are CLI-configurable (relative or absolute).
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import h5py
import numpy as np
import scipy.io as sio
import torch
from scipy import ndimage

from Train_Fusion import GuidedVisionTransformerT
from VisionTransformer_7_RSD import VisionTransformer as ViT_RSD
from VisionTransformer_7_T import VisionTransformer as ViT_T


def add_zero_padding(x, margin=3):
    """Pad (H, W, C) with zeros."""
    out = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2 * margin, x.shape[2]), dtype=x.dtype)
    out[margin : x.shape[0] + margin, margin : x.shape[1] + margin, :] = x
    return out


def smooth_segmentation_by_connectivity(label_map, num_classes, min_region_size=90, neighbor_radius=9):
    h, w = label_map.shape
    label_map = label_map.astype(np.int32).copy()
    for cls in range(1, num_classes + 1):
        mask = label_map == cls
        if not mask.any():
            continue
        labeled, num_comp = ndimage.label(mask)
        if num_comp == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        small_ids = [i for i, s in enumerate(sizes) if i != 0 and s < min_region_size]
        for comp_id in small_ids:
            ys, xs = np.where(labeled == comp_id)
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y - neighbor_radius), min(h, y + neighbor_radius + 1)
                x0, x1 = max(0, x - neighbor_radius), min(w, x + neighbor_radius + 1)
                window = label_map[y0:y1, x0:x1].reshape(-1)
                window = window[window != 0]
                if window.size == 0:
                    continue
                label_map[y, x] = np.bincount(window).argmax()
    return label_map


def find_latest_model(model_dir, prefix):
    if not os.path.isdir(model_dir):
        return None
    files = [f for f in os.listdir(model_dir) if f.startswith(prefix) and f.endswith(".pth")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, files[0])


def load_mat_array(path, key=None):
    """Load an array from MATLAB v7.3 (h5py) or classic .mat."""
    try:
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            use_key = key if key in f else keys[0]
            return np.array(f[use_key]), use_key
    except OSError:
        mat = sio.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        use_key = key if key in mat else keys[0]
        return np.array(mat[use_key]), use_key


def parse_args():
    p = argparse.ArgumentParser(description="PUS-ViT full-scene inference")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/barnaul")
    p.add_argument("--pgcl_ckpt", type=str, default=None)
    p.add_argument("--fusion_ckpt", type=str, default=None)
    p.add_argument("--t_mat", type=str, required=True, help="Full-scene T-matrix .mat")
    p.add_argument("--t_key", type=str, default=None)
    p.add_argument("--gt_mat", type=str, default=None, help="Optional GT map (0=background)")
    p.add_argument("--gt_key", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--num_classes", type=int, default=9)
    p.add_argument("--num_topics", type=int, default=15)
    p.add_argument("--window_size", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--max_rows", type=int, default=None, help="Optional row cap for smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    pgcl_path = args.pgcl_ckpt or find_latest_model(args.ckpt_dir, "pgcl_model_best_")
    fusion_path = args.fusion_ckpt or find_latest_model(args.ckpt_dir, "fusion_model_best_")
    if not pgcl_path or not os.path.isfile(pgcl_path):
        raise SystemExit(f"Stage-1 checkpoint not found under {args.ckpt_dir}")
    if not fusion_path or not os.path.isfile(fusion_path):
        raise SystemExit(f"Stage-2 checkpoint not found under {args.ckpt_dir}")
    print(f"Stage-1 ckpt: {pgcl_path}")
    print(f"Stage-2 ckpt: {fusion_path}")

    rsd_model = ViT_RSD(
        embed_dim=64,
        num_classes=args.num_classes,
        num_topics=args.num_topics,
        use_pgcl=True,
        use_bow=True,
        in_channels=12,
        depth=3,
        mlp_dim=256,
        patch_size=7,
    ).to(device)
    rsd_model.load_state_dict(torch.load(pgcl_path, map_location=device), strict=False)
    rsd_model.eval()
    guidance_topic = copy.deepcopy(rsd_model.topic_model).to(device).eval()
    guidance_pgcl = copy.deepcopy(rsd_model.pgcl).to(device).eval()
    for p in list(guidance_topic.parameters()) + list(guidance_pgcl.parameters()):
        p.requires_grad = False

    base_model = ViT_T(
        embed_dim=64,
        num_classes=args.num_classes,
        num_topics=args.num_topics,
        use_pgcl=False,
        rsd_model_path=pgcl_path,
        in_channels=10,
        depth=4,
        mlp_dim=256,
        patch_size=7,
    ).to(device)
    fusion_model = GuidedVisionTransformerT(
        base_model, guidance_topic, guidance_pgcl, num_classes=args.num_classes
    ).to(device)
    fusion_model.load_state_dict(torch.load(fusion_path, map_location=device), strict=False)
    fusion_model.eval()

    t_arr, t_key = load_mat_array(args.t_mat, args.t_key)
    print(f"Loaded T cube key='{t_key}', shape={t_arr.shape}")
    if t_arr.ndim != 3:
        raise SystemExit(f"Expected 3-D T cube, got shape {t_arr.shape}")
    if t_arr.shape[0] in (9, 10):
        data_hwi = np.transpose(t_arr, (1, 2, 0))
    elif t_arr.shape[-1] in (9, 10):
        data_hwi = t_arr
    else:
        data_hwi = np.transpose(t_arr, (2, 1, 0))
    height, width, channels = data_hwi.shape
    if channels not in (9, 10):
        print(f"Warning: channel dim={channels}; expected 9 or 10.")

    if args.gt_mat:
        gt_arr, gt_key = load_mat_array(args.gt_mat, args.gt_key)
        print(f"Loaded GT key='{gt_key}', shape={gt_arr.shape}")
        if gt_arr.ndim == 2 and gt_arr.shape != (height, width):
            gt_arr = gt_arr.T
        data_gt = gt_arr.astype(np.int32)
    else:
        data_gt = np.ones((height, width), dtype=np.int32)

    margin = (args.window_size - 1) // 2
    padded = add_zero_padding(data_hwi, margin=margin)
    outputs = np.zeros((height, width), dtype=np.int32)
    row_limit = height if args.max_rows is None else min(height, args.max_rows)

    print(f"Infer {height}x{width} (rows 0..{row_limit - 1}), window={args.window_size}, device={device}")
    with torch.no_grad():
        for i in range(row_limit):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  row {i + 1}/{row_limit}")
            for j in range(width):
                if int(data_gt[i, j]) == 0:
                    continue
                patch = padded[i : i + args.window_size, j : j + args.window_size, :]
                xt = torch.FloatTensor(patch.transpose(2, 0, 1)[None]).to(device)
                fusion_logits, _ = fusion_model(xt)
                outputs[i, j] = int(torch.argmax(fusion_logits, dim=1).item()) + 1

    raw_path = os.path.join(args.output_dir, "result_full_raw.mat")
    sio.savemat(raw_path, {"output": outputs})
    print(f"Saved {raw_path}")

    if args.postprocess:
        smoothed = smooth_segmentation_by_connectivity(outputs, num_classes=args.num_classes)
        post_path = os.path.join(args.output_dir, "result_full_post.mat")
        sio.savemat(post_path, {"output": smoothed})
        print(f"Saved {post_path}")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
