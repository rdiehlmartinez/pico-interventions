"""
Checkpointing Config

Specifies the hyperparameters for the checkpointing process; checkpointing is used to save
the model and optimizer states, as well as the learning dynamics metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from ._constants import (
    RUNS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    FABRIC_CHECKPOINT_DIR,
    FABRIC_CHECKPOINT_FILENAME,
    EVAL_RESULTS_DIR,
    LEARNING_DYNAMICS_DIR,
)


@dataclass
class TrainingCheckpointingConfig:
    auto_resume: bool = True


@dataclass
class EvaluationCheckpointingConfig:
    eval_results_dir: str = EVAL_RESULTS_DIR


@dataclass
class LearningDynamicsCheckpointingConfig:
    # Suffixes of the layers to compute learning dynamics for
    layer_suffixes: List[str] = field(
        default_factory=lambda: [
            "attention.v_proj",
            "attention.o_proj",
            "swiglu.w_2",
        ]
    )

    # Sequence index at which to extract hidden states; by default, we extract the hidden states
    # at the last token of the sequence (-1)
    sequence_idx: int = -1

    # size of the sub-batch used for extracting learning dynamics states
    batch_size: int = 8

    # Path to evaluation dataset - used across learning dynamics checkpointing for consistency
    # NOTE: set to None to disable extracting learning dynamics states for an eval_batch
    # NOTE: this dataset should be small, ideally just a batch of additional data
    eval_data: Optional[str] = "pico-lm/pretokenized-paloma-tinsy"


@dataclass
class HuggingFaceCheckpointingConfig:
    # Should be in the format of <(username or organization name)>/<repo_name>, e.g. pico-lm/demo
    repo_id: str = ""

    # HuggingFace Collection Slug (specifies a tag for the run)
    collection_slug: Optional[str] = None


@dataclass
class CheckpointingConfig:
    # Name of the run
    run_name: Optional[str] = None

    runs_dir: str = RUNS_DIR
    checkpoints_dir: str = CHECKPOINTS_DIR
    logs_dir: str = LOGS_DIR
    fabric_checkpoint_dir: str = FABRIC_CHECKPOINT_DIR
    fabric_checkpoint_filename: str = FABRIC_CHECKPOINT_FILENAME
    learning_dynamics_dir: str = LEARNING_DYNAMICS_DIR

    # How often to save checkpoints
    save_every_n_steps: int = 1000

    # Whether to save checkpoints to HuggingFace
    save_to_hf: Optional[bool] = False
    hf_checkpoint: HuggingFaceCheckpointingConfig = field(
        default_factory=HuggingFaceCheckpointingConfig
    )

    training: TrainingCheckpointingConfig = field(
        default_factory=TrainingCheckpointingConfig
    )
    evaluation: EvaluationCheckpointingConfig = field(
        default_factory=EvaluationCheckpointingConfig
    )
    learning_dynamics: LearningDynamicsCheckpointingConfig = field(
        default_factory=LearningDynamicsCheckpointingConfig
    )
