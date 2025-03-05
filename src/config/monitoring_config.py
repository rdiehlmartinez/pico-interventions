"""
Monitoring Config

Specifies the monitoring process, e.g. how to log metrics and keep track of training progress.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_every_n_steps: int = 100


@dataclass
class WandbConfig:
    project: Optional[str] = "pico"
    entity: Optional[str] = "pico-lm"


@dataclass
class MonitoringConfig:
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Weights and Biases
    save_to_wandb: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)
