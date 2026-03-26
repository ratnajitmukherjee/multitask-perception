"""File that defines the checkpointer class."""
import logging
import os
from typing import Any
from typing import Optional

import torch
import logging
logger = logging.getLogger(__name__)
from torch.nn.parallel import DistributedDataParallel

# TODO: app_settings not ported — from modular_training_framework.app_settings import retrieve_settings
# TODO: global_settings not ported — from modular_training_framework.global_settings import DEFAULT_ARTIFACT_FILE_NAME
# TODO: default_config not ported — from modular_training_framework.od.default_config import cfg
# TODO: clearml integration not ported — from modular_training_framework.od.utils.clearml import retrieve_model_artifact_from_s3
# TODO: clearml integration not ported — from modular_training_framework.od.utils.clearml import save_model_to_s3


class CheckPointer:
    """Checkpointer Class."""

    _last_checkpoint_name: str = "last_checkpoint"  # TODO: was DEFAULT_ARTIFACT_FILE_NAME from modular_training_framework.global_settings

    def __init__(
        self,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        save_dir: str = "",
        save_to_disk: bool = None,
        logger: Any = None,
        run_folder_name: str = None,
    ) -> None:
        """Init function for the checkpointer class.
        Args:
            model: Structure of the defined model
            optimizer: Optimizer used for the run
            scheduler: Scheduler used for the run
            save_dir: Save directory for the checkpoint
            save_to_disk: Whether to save to disk or not
            logger: logger
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.run_folder_name = run_folder_name

    def save(self, name: str, save_model_to_s3_bucket: bool = False, **kwargs) -> None:
        """Function to save the checkpoint.
        Args:
            name: Save filename of the checkpoint
            save_model_to_s3_bucket: Boolean to save to S3 or not
            **kwargs:
        """
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        if isinstance(self.model, DistributedDataParallel):
            data["model"] = self.model.module.state_dict()
        else:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info(f"Saving local checkpoint to {save_file}")
        torch.save(data, save_file)
        if save_model_to_s3_bucket:
            self.logger.info(f"Uploading saved checkpoint to S3...")
            save_model_to_s3(
                save_file, self.run_folder_name, model_file_name=f"{name}.pth"
            )

        self.tag_last_checkpoint(save_file)

    def load(
        self,
        f: str = None,
        use_latest: bool = True,
        s3_model_version: Optional[str] = None,
        device: Any = "cpu",
    ) -> Any:
        """Function to load a checkpoint from local or using ClearML.
        Args:
            f: Filepath of the checkpoint (local)
            use_latest: Boolean to use latest version
            load_model_from_s3: Boolean to download model from S3
            device: Device cpu/gpu

        Returns:
            The loaded checkpoint
        """
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}

        if s3_model_version:
            logger.info(f"Loading checkpoint from S3: {s3_model_version}")
            checkpoint = retrieve_model_artifact_from_s3(model_version=s3_model_version)
        else:
            logger.info(f"Loading local checkpoint from {f}")
            checkpoint = self._load_file(f, device=device)
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = self.model.module

        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info(f"Loading optimizer from {f}")
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info(f"Loading scheduler from {f}")
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self) -> str:
        """Function to return the filepath of the checkpoint file.
        Returns:
            The filepath of the checkpoint file
        """
        save_file = os.path.join(cfg.OUTPUT_DIR, self._last_checkpoint_name)

        try:
            with open(save_file) as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except OSError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def has_checkpoint(self) -> bool:
        """Function to verify if checkpoint exists at the checkpoint path.
        Returns:
            True if the checkpoint exists
        """
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename: str) -> None:
        """Function to tag the checkpoint
        Args:
            last_filename: Filename to be tagged
        """
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f: str, device: Any = "cpu") -> Any:
        """Function to load file from filepath f.
        Args:
            f: Filepath of the checkpoint.
            device: Device cpu/gpu

        Returns:
            The loaded checkpoint
        """
        return torch.load(f, map_location=torch.device(device))
