#!/usr/bin/env python3
import argparse
import logging
import os
import glob
import yaml

import torch
import torch.utils.data

from multitask_perception.engine.inference import do_evaluation
from multitask_perception.modeling.build import build_model
from multitask_perception.utils import dist_util
from multitask_perception.utils.checkpoint import CheckPointer
from multitask_perception.utils.dist_util import synchronize
from multitask_perception.utils.flops_counter import get_model_complexity_info
from multitask_perception.utils.energy_meter import EnergyMeter
from contextlib import ExitStack
from multitask_perception.config import get_cfg_defaults, sub_cfg_dict

cfg = get_cfg_defaults()


def setup_logger(name, rank, output_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if rank == 0:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if output_dir:
            fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def evaluation(cfg, ckpt, eval_only, calc_energy, distributed, precision_display, iou_threshold, dataset_type, args_outputdir):
    logger = logging.getLogger("Object Detection.inference")

    model = build_model(cfg)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)

    if not eval_only:
        image_res = (3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
        flops_count, params_count = get_model_complexity_info(model, image_res)
        print("MAC Count:", flops_count)
        print("Number of Parameters:", params_count)

    if not cfg.OUTPUT_DIR:
        if args_outputdir:
            cfg.defrost()
            cfg.OUTPUT_DIR = args_outputdir
    with EnergyMeter(dir=cfg.OUTPUT_DIR) if calc_energy else ExitStack():
        inf_results, eval_results = do_evaluation(cfg, model, distributed, get_inf_time=False if eval_only else True,
                                                  precision_display=precision_display,
                                                  iou_threshold=iou_threshold,
                                                  dataset_type=dataset_type)

    if not eval_only:
        for i, dataset_name in enumerate(cfg.DATASETS.TEST):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            list_of_files = glob.glob(output_folder + "**/*.txt")  # * means all if need specific format then *.csv
            result_path = max(list_of_files, key=os.path.getmtime)

            result_strings = []
            result_strings.append("\nInference Speed (in FPS): {:.4f}".format(1000 / inf_results[i]))
            result_strings.append("MAC Count (in GMac): {}".format(flops_count))
            result_strings.append("Number of Parameters (in M): {}".format(params_count))
            with open(result_path, "a") as f:
                f.write('\n'.join(result_strings))

def main():
    parser = argparse.ArgumentParser(description='Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
        required=True
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument("--output_dir", default="eval_results", type=str, help="The directory to store evaluation results.")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--eval_only", action='store_true',
                        help='If set, outputs MAP without other statistics such as Inference time, Energy and Number of Parameters')
    parser.add_argument("--calc_energy", action='store_true',
                        help='If set, measures and outputs the Energy consumption')
    parser.add_argument("--precision_display", action="store_true", help="If set, display precision, recall and F-score")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="The default IOU threshold for precision display")
    parser.add_argument("--dataset_type", default="", type=str, help="Specify dataset type. Currently support voc and coco.")
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Merge head-specific sub-config if detection is enabled
    raw_cfg = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
    tasks = raw_cfg.get("TASK", {}).get("ENABLED", ["detection"])
    if "detection" in tasks:
        det_head = raw_cfg["MODEL"]["HEADS"]["DETECTION"]["NAME"]
        if det_head in sub_cfg_dict:
            cfg.merge_from_other_cfg(sub_cfg_dict[det_head])
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("Object Detection", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    if args.precision_display and args.dataset_type == "":
        logger.info('Precision display argument requires dataset type')
        return

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt,
               eval_only=args.eval_only,
               calc_energy=args.calc_energy,
               distributed=distributed,
               precision_display=args.precision_display,
               iou_threshold=args.iou_threshold,
               dataset_type=args.dataset_type,
               args_outputdir=args.output_dir)


if __name__ == '__main__':
    main()
