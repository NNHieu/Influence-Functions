from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from lightning_utilities.core.imports import RequirementCache
from lightning_lite.utilities.types import _PATH
from torch import Tensor

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

try:
    import wandb
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run
except ModuleNotFoundError:
    # needed for test mocks, these tests shall be updated
    wandb, Run, RunDisabled = None, None, None

_WANDB_AVAILABLE = RequirementCache("wandb")
_WANDB_GREATER_EQUAL_0_10_22 = RequirementCache("wandb>=0.10.22")
_WANDB_GREATER_EQUAL_0_12_10 = RequirementCache("wandb>=0.12.10")


def _scan_checkpoints(
    checkpoint_callback: Checkpoint, logged_model_time: dict
) -> List[Tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.
    """

    # get checkpoints to be saved with associated score
    checkpoints = dict()
    if hasattr(checkpoint_callback, "last_model_path") and hasattr(
        checkpoint_callback, "current_score"
    ):
        checkpoints[checkpoint_callback.last_model_path] = (
            checkpoint_callback.current_score,
            ("latest",),
        )

    if hasattr(checkpoint_callback, "best_model_path") and hasattr(
        checkpoint_callback, "best_model_score"
    ):
        checkpoints[checkpoint_callback.best_model_path] = (
            checkpoint_callback.best_model_score,
            ("best",),
        )

    # print(checkpoints)
    if hasattr(checkpoint_callback, "best_k_models"):
        for key, value in checkpoint_callback.best_k_models.items():
            # print(key)
            if key in checkpoints:
                checkpoints[key] = (value, checkpoints[key][1])
            else:
                checkpoints[key] = (value, None)

    checkpoints = sorted(
        (Path(p).stat().st_mtime, p, s, tag)
        for p, (s, tag) in checkpoints.items()
        if Path(p).is_file()
    )
    checkpoints = [
        c
        for c in checkpoints
        if c[1] not in logged_model_time.keys() or logged_model_time[c[1]] < c[0]
    ]
    if len(checkpoints) > 0:
        latest_tag = checkpoints[-1][3]
        if latest_tag is None:
            latest_tag =  ("latest",)
        else:
            latest_tag += ("latest",)
        checkpoints[-1] = (*checkpoints[-1][0:3], latest_tag) # Ughhhhhh
    return checkpoints


class CustomWandbLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        artifact_prefix: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: str = "lightning_logs",
        log_model: Union[str, bool] = False,
        experiment: Union[Run, RunDisabled, None] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        if log_model == "all":
            raise NotImplemented
        self.artifact_prefix = artifact_prefix
        super().__init__(
            name,
            save_dir,
            version,
            offline,
            dir,
            id,
            anonymous,
            project,
            log_model,
            experiment,
            prefix,
            **kwargs,
        )

    # @rank_zero_only
    # def finalize(self, status: str) -> None:
    #     if status != "success":
    #         # Currently, checkpoints only get logged on success
    #         return
    #     # log checkpoints as artifacts
    #     if self._checkpoint_callback and self._experiment is not None:
    #         self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: Checkpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)
        print(checkpoints)
        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = (
                {
                    "score": s.item() if isinstance(s, Tensor) else s,
                    "original_filename": Path(p).name,
                    checkpoint_callback.__class__.__name__: {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                        ]
                        # ensure it does not break if `ModelCheckpoint` args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
                if _WANDB_GREATER_EQUAL_0_10_22
                else None
            )
            if self.artifact_prefix is not None:
                name = f"model-{self.artifact_prefix}-{self.experiment.id}"
            else:
                name = f"model-{self.experiment.id}"
            # name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(
                name=name, type="model", metadata=metadata
            )
            print(artifact)
            artifact.add_file(p, name="model.ckpt")
            self.experiment.log_artifact(artifact, aliases=tag)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t

    def after_save_checkpoint(self, checkpoint_callback: Checkpoint) -> None:
        # log checkpoints as artifacts
        if (
            self._log_model == "all"
            or self._log_model is True
            and hasattr(checkpoint_callback, "save_top_k")
            and checkpoint_callback.save_top_k == -1
        ):
            raise NotImplemented
            # self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            if (
                hasattr(checkpoint_callback, "avail_to_wandb")
                and not checkpoint_callback.avail_to_wandb
            ):
                return
            self._checkpoint_callback = checkpoint_callback

    # def _scan_and_log_checkpoints(self, checkpoint_callback: Checkpoint) -> None:
    #     if (
    #         hasattr(checkpoint_callback, "avail_to_wandb")
    #         and not checkpoint_callback.avail_to_wandb
    #     ):
    #         return
    #     return super()._scan_and_log_checkpoints(checkpoint_callback)
