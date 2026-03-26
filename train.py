#!/usr/bin/env python3
import argparse
import logging
import os
import yaml

import torch
import torch.distributed as dist
from numpy import random
from multitask_perception.engine.inference import do_evaluation
from multitask_perception.data.build import make_data_loader
from multitask_perception.engine.trainer import do_train
from multitask_perception.modeling.build import build_model
from multitask_perception.solver import make_optimizer
from multitask_perception.solver.lr_scheduler import make_lr_scheduler
from multitask_perception.utils import dist_util, mkdir
from multitask_perception.utils.checkpoint import CheckPointer
from multitask_perception.utils.dist_util import synchronize
from multitask_perception.config import get_cfg_defaults, sub_cfg_dict

cfg = get_cfg_defaults()


def str2bool(s):
    return s.lower() in ("true", "1", "yes")


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


def train(cfg, args):
    logger = logging.getLogger(cfg.LOGGER.NAME + ".trainer")
    model = build_model(cfg)

    if args.distributed:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device = torch.device("cuda:{}".format(args.local_rank))
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    if args.pretrained_path != "":
        logger.info("pretrained model found.")
        from torch.nn.parallel import DistributedDataParallel

        checkpoint = torch.load(args.pretrained_path, map_location=torch.device("cpu"))
        if isinstance(model, DistributedDataParallel):
            model_load = model.module
        else:
            model_load = model

        model_dict = model_load.state_dict()
        pretrained_dict = {
            k: v
            for k, v in checkpoint["model"].items()
            if k in model_dict and (model_dict[k].shape == checkpoint["model"][k].shape)
        }
        model_dict.update(pretrained_dict)

        model_load.load_state_dict(model_dict)

    checkpointer = CheckPointer(
        model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(
        cfg,
        is_train=True,
        distributed=args.distributed,
        max_iter=max_iter,
        start_iter=arguments["iteration"],
    )

    model = do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        args,
    )
    return model


def main():
    # arguments
    # any New config should be added  to config file and you  pass this config file at the arguments
    parser = argparse.ArgumentParser(
        description="Single Shot MultiBox Detector Training With PyTorch"
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        required=True,
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--log_step", default=10, type=int, help="Print logs every log_step"
    )
    parser.add_argument(
        "--save_step", default=2500, type=int, help="Save checkpoint every save_step"
    )
    parser.add_argument(
        "--eval_step",
        default=2500,
        type=int,
        help="Evaluate dataset every eval_step, disabled when eval_step < 0",
    )
    parser.add_argument(
        "--pretrained_path",
        default="",
        type=str,
        help="pretrianed path , default is empty, if you put any path it will pretrain the model",
    )
    parser.add_argument("--use_tensorboard", default=True, type=str2bool)
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for processes. Seed must be fixed for distributed training",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--calc_energy",
        action="store_true",
        help="If set, measures and outputs the Energy consumption",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--precision_display",
        action="store_true",
        help="If set, display precision and recall",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="The default IOU threshold for precision display",
    )
    parser.add_argument(
        "--dataset_type",
        default="",
        type=str,
        help="Specify dataset type. Currently support voc and coco.",
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus
    # remove torch.backends.cudnn.benchmark to avoid potential risk

    if args.distributed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=num_gpus,
            rank=args.local_rank,
        )
        synchronize()

    # defined by 'head' not meta_architecture
    det_head = yaml.load(open(args.config_file), Loader=yaml.FullLoader)["MODEL"][
        "HEAD"
    ]["DET_NAME"]
    sub_cfg = sub_cfg_dict[det_head]

    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger(cfg.LOGGER.NAME, dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    if args.precision_display and args.dataset_type == "":
        logger.info("Precision display argument requires dataset type")
        return

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info("Start evaluating...")
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(
            cfg,
            model,
            distributed=args.distributed,
            precision_display=args.precision_display,
            iou_threshold=args.iou_threshold,
            dataset_type=args.dataset_type,
        )


if __name__ == "__main__":
    main()
