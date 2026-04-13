#!/usr/bin/env python3

import argparse
import glob
import json
import os
import pathlib

import numpy as np
import torch
import yaml
from PIL import Image

from multitask_perception.config import get_cfg_defaults, sub_cfg_dict
from multitask_perception.data.transforms import build_transforms
from multitask_perception.modeling.build import build_model
from multitask_perception.structures.container import Container
from multitask_perception.utils.checkpoint import CheckPointer
from multitask_perception.utils.output_processor import convert_to_format

# ---- Default constants ----
DEFAULT_CFG = ""
DEFAULT_CKPT = None
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_DATASET_TYPE = "coco"
DEFAULT_INPUT_DIR = "./input"
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_OUTPUT_FORMAT = "img"
SUPPORTED_IMAGE_FORMATS = [".png", ".PNG", ".jpg", ".JPG"]


def main():
    parser = argparse.ArgumentParser(description="Inference application.")
    parser.add_argument("--config-file", type=str, default=DEFAULT_CFG, metavar="FILE",
                        required=True, help="Path to config file.")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                        help="Trained weights (default: %(default)s).")
    parser.add_argument("--score_threshold", type=restricted_float, default=DEFAULT_SCORE_THRESHOLD,
                        help="Threshold value (default: %(default)s).")
    parser.add_argument("--dataset_type", type=str, choices=['coco', 'voc'],
                        default=DEFAULT_DATASET_TYPE,
                        help='Specify dataset type (default: %(default)s).')
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR,
                        help='Input directory/image to be predicted (default: %(default)s).')
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory that will contain the prediction(s) (default: %(default)s).')
    parser.add_argument("--output_format", type=str, choices=['img', 'json'],
                        default=DEFAULT_OUTPUT_FORMAT,
                        help='Output format (default: %(default)s).')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose mode.')
    parser.add_argument("config_options", nargs=argparse.REMAINDER,
                        help="Configuration options that overwrites those from the configuration file.")
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # ---- Load config ----
    cfg = get_cfg_defaults()

    # Merge head-specific sub-config if detection is enabled
    raw_cfg = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
    tasks = raw_cfg.get("TASK", {}).get("ENABLED", ["detection"])
    if "detection" in tasks:
        det_head = raw_cfg["MODEL"]["HEADS"]["DETECTION"]["NAME"]
        if det_head in sub_cfg_dict:
            cfg.merge_from_other_cfg(sub_cfg_dict[det_head])

    cfg.merge_from_file(args.config_file)
    if args.config_options:
        cfg.merge_from_list(args.config_options)
    cfg.freeze()

    if args.verbose:
        print("*" * 80)
        print(f"Loaded configuration file {args.config_file}")
        print(f"Running with config:\n{cfg}")
        print("*" * 80)

    # ---- Build and load model ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.to(device)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(args.ckpt, use_latest=args.ckpt is None)
    model.eval()

    transforms = build_transforms(cfg, is_train=False)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.verbose:
        print(f"\nModel loaded. Device: {device}")

    # ---- Run inference ----
    input_path = args.input
    if os.path.isdir(input_path):
        image_paths = find_files_with_extensions(input_path, SUPPORTED_IMAGE_FORMATS)
        if not image_paths:
            print(f"Warning: no images found with extensions {SUPPORTED_IMAGE_FORMATS}")
            return
        if args.verbose:
            print(f"\nRunning inference on {len(image_paths)} images from {input_path}:")
        for i, img_path in enumerate(image_paths):
            if args.verbose:
                print(f"({i + 1:04d}/{len(image_paths):04d}) {os.path.basename(img_path)}")
            try:
                infer_single_image(
                    img_path, model, transforms, device, cfg, args
                )
            except OSError as e:
                print(f"ERROR: {img_path}: {e}")
    else:
        if args.verbose:
            print(f"\nRunning inference on {input_path}:")
        infer_single_image(input_path, model, transforms, device, cfg, args)


@torch.no_grad()
def infer_single_image(image_path, model, transforms, device, cfg, args):
    """Run inference on a single image and save the result."""
    image = np.array(Image.open(image_path).convert("RGB"))
    image_basename = os.path.basename(image_path)
    height, width = image.shape[:2]
    image_preprocessed = transforms(image)[0].unsqueeze(0)

    enabled = cfg.TASK.ENABLED
    has_det = "detection" in enabled
    has_seg = "segmentation" in enabled
    det_result = None
    seg_result = None

    if has_det and has_seg:
        # Multitask
        detections, segmentations = model(image_preprocessed.to(device))
        boxes, labels, scores = process_detections(detections, args.score_threshold)
        seg_mask, seg_class_ids = process_segmentations(segmentations, width, height)
        det_result, seg_result = convert_to_format(
            fmt=args.output_format, image=image,
            boxes=boxes, labels=labels, scores=scores,
            seg_mask=seg_mask, seg_class_ids=seg_class_ids,
            det_class_names=None, seg_class_names=None,
        )
    elif has_det:
        # Detection only
        output = model(image_preprocessed.to(device))
        detections = output[0] if isinstance(output, (list, tuple)) else output
        boxes, labels, scores = process_detections(detections, args.score_threshold)
        det_result, _ = convert_to_format(
            fmt=args.output_format, image=image,
            boxes=boxes, labels=labels, scores=scores,
            seg_mask=None, seg_class_ids=None,
            det_class_names=None, seg_class_names=None,
        )
    elif has_seg:
        # Segmentation only
        segmentations = model(image_preprocessed.to(device))
        seg_mask, seg_class_ids = process_segmentations(segmentations, width, height)
        _, seg_result = convert_to_format(
            fmt=args.output_format, image=image,
            boxes=None, labels=None, scores=None,
            seg_mask=seg_mask, seg_class_ids=seg_class_ids,
            det_class_names=None, seg_class_names=None,
        )
    else:
        raise NotImplementedError(f"No recognized tasks in TASK.ENABLED: {enabled}")

    # Save result
    save_result(args.output_dir, args.output_format, image_basename, det_result, seg_result)


def process_detections(detections, score_threshold):
    """Extract boxes, labels, scores from detection output."""
    if isinstance(detections, Container):
        detections = detections.to(torch.device("cpu")).numpy()
    boxes, labels, scores = detections["boxes"], detections["labels"], detections["scores"]
    indices = scores > score_threshold
    boxes = boxes[indices]
    labels = labels[indices].astype(int)
    scores = scores[indices]
    return boxes, labels, scores


def process_segmentations(segmentations, width, height):
    """Convert raw segmentation output to mask and class ids."""
    seg_mask_gpu = torch.argmax(segmentations.squeeze(dim=0), dim=0)
    seg_mask = seg_mask_gpu.cpu().numpy().squeeze().astype(np.uint8)
    try:
        seg_class_ids = seg_mask_gpu.unique().cpu().tolist()
    except RuntimeError:
        seg_class_ids = list(set(seg_mask.flatten().tolist()))
    seg_mask = Image.fromarray(seg_mask)
    seg_mask = seg_mask.resize(size=(width, height), resample=Image.NEAREST)
    return seg_mask, seg_class_ids


def save_result(output_dir, output_format, image_basename, det_result, seg_result):
    """Save inference result to disk."""
    if output_format == "img":
        result = seg_result if seg_result is not None else det_result
        if result is not None:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(os.path.join(output_dir, image_basename))
    elif output_format == "json":
        output_dict = {"detection": det_result, "segmentation": seg_result}
        json_path = os.path.join(output_dir, os.path.splitext(image_basename)[0] + ".json")
        with open(json_path, "w") as f:
            json.dump(output_dict, f, sort_keys=True)


def find_files_with_extensions(where, extensions):
    """Find all files matching given extensions in a directory."""
    image_paths = []
    for ext in extensions:
        image_paths += glob.glob(os.path.join(where, "*" + ext))
    return sorted(image_paths)


def restricted_float(val):
    try:
        val = float(val)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{val} not a floating-point literal')
    if val < 0.0 or val > 1.0:
        raise argparse.ArgumentTypeError(f'{val} not in range [0.0, 1.0]')
    return val


if __name__ == '__main__':
    main()
