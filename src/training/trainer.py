"""
Pico Language Model Trainer

This Trainer implements a minimalistic end-to-end training pipeline of the Pico language model with
distributed training support via Lightning Fabric. It provides a modular and configurable training
pipeline with the features:

    - Configuration Management: YAML-based configuration for all aspects of training
    - Distributed Training: Multi-GPU support via Lightning Fabric
    - Checkpointing: Regular model saving and training state recovery
    - Evaluation: Periodic model evaluation on validation datasets
    - Logging: Comprehensive metric tracking and experiment monitoring
    - Optimization: Support for gradient accumulation, clipping, and LR scheduling
"""

import logging
import lightning as L
import torch
import torch.nn.functional as F
import os
import platform
import psutil
import yaml
from lightning.fabric.utilities.rank_zero import rank_zero_only

from datasets import Dataset, load_dataset
from typing import Dict, Any

from src.training.utils import (
    initialize_run_dir,
    initialize_fabric,
    initialize_configuration,
    initialize_dataset,
    initialize_tokenizer,
    initialize_dataloader,
    initialize_lr_scheduler,
    initialize_hf_checkpointing,
    initialize_wandb,
    initialize_logging,
    initialize_optimizer,
    initialize_model,
)
from src.training.utils.logging import pretty_print_yaml_config
from src.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_evaluation_results,
    compute_learning_dynamics_states,
    save_learning_dynamics_states,
)

from src.evaluation import run_evaluation


class Trainer:
    def __init__(self, config_path: str):
        """
        Initializes the Trainer class. This Trainer class implements a `train` method, which is the
        main entry point for training the Pico model. Before calling `train`, the Trainer class
        initializes the following:

            - Configuration loading and validation
            - Model, optimizer, and dataset setup
            - Logging and experiment tracking setup
            - Checkpoint management

        Args:
            config_path (str): Path to the YAML configuration file containing any overrides.
        """

        ########################################################
        #
        # Basic Initialization of Configs, Fabric, Model, Optimizer, etc.
        #
        ########################################################

        # Setup Config
        self.configs = initialize_configuration(config_path)

        # Setup Run Directory (i.e. where we store checkpoints, logs, etc.)
        initialize_run_dir(checkpointing_config=self.configs["checkpointing"])

        # Setup Logger
        if self.configs["monitoring"].save_to_wandb:
            wandb_logger = initialize_wandb(
                monitoring_config=self.configs["monitoring"],
                checkpointing_config=self.configs["checkpointing"],
            )
        else:
            wandb_logger = None

        # Setup Fabric
        self.fabric = initialize_fabric(
            training_config=self.configs["training"],
            wandb_logger=wandb_logger,
        )
        L.seed_everything(42, verbose=False)

        # Set up logging
        self.logger = initialize_logging(
            monitoring_config=self.configs["monitoring"],
            checkpointing_config=self.configs["checkpointing"],
            fabric=self.fabric,
        )

        # Setup Model, Optimizer, and Dataloaders
        self.model = initialize_model(model_config=self.configs["model"])
        self.optimizer = initialize_optimizer(
            training_config=self.configs["training"], model=self.model
        )
        self.lr_scheduler = initialize_lr_scheduler(
            training_config=self.configs["training"], optimizer=self.optimizer
        )

        # Wrap with Fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Setup HuggingFace Checkpointing
        if self.configs["checkpointing"].save_to_hf:
            initialize_hf_checkpointing(
                checkpointing_config=self.configs["checkpointing"], fabric=self.fabric
            )

        ########################################################
        #
        # Boilerplate to deal with loading/resuming from checkpoints
        #
        ########################################################

        self.should_load_checkpoint = self.configs["checkpointing"].training.auto_resume

        # Possibly load a checkpoint
        if self.should_load_checkpoint:
            resume_checkpoint = load_checkpoint(
                checkpointing_config=self.configs["checkpointing"],
                checkpoint_step="latest",
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
            )

            if resume_checkpoint:
                (
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.initial_batch_step,
                ) = resume_checkpoint
            else:
                self.initial_batch_step = 0
        else:
            self.initial_batch_step = 0

        ########################################################
        #
        # Initialization of Dataset & DataLoader (possibly fast-forwarding to correct batch)
        #
        ########################################################

        self.train_dataset, fast_forward_steps = initialize_dataset(
            data_config=self.configs["data"],
            fabric=self.fabric,
            initial_batch_step=self.initial_batch_step,
            return_fast_forward_steps=True,
        )

        self.train_dataloader = initialize_dataloader(
            data_config=self.configs["data"],
            training_config=self.configs["training"],
            fabric=self.fabric,
            dataset=self.train_dataset,
        )
        self.train_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, use_distributed_sampler=False
        )

        self.tokenizer = initialize_tokenizer(data_config=self.configs["data"])

        # NOTE: We may need to fast-forward the iterator to the correct step so that we can
        # continue from the correct batch of data we would have seen had training not
        # previously stopped.
        train_iterator = iter(self.train_dataloader)
        if fast_forward_steps > 0:
            fast_forward_sub_steps = (
                fast_forward_steps
                * self.configs["training"].optimization.gradient_accumulation_steps
            )
            for _ in range(fast_forward_sub_steps):
                next(train_iterator)

        self.train_iterator = train_iterator

        # NOTE: Sychronizing processes after fast-forwarding iterator
        self.fabric.barrier()

        ########################################################
        #
        # Helper flags used during training for checkpointing and evaluation
        #
        ########################################################

        # Helper flag to determine if we should evaluate the model
        self.should_evaluate = (
            self.configs["evaluation"].metrics is not None
            and len(self.configs["evaluation"].metrics) > 0
        )

        self.should_compute_learning_dynamics = (
            self.configs["checkpointing"].learning_dynamics.layer_suffixes is not None
            and len(self.configs["checkpointing"].learning_dynamics.layer_suffixes) > 0
        )

        if self.should_compute_learning_dynamics:
            if self.configs["checkpointing"].learning_dynamics.eval_data is not None:
                self.learning_dynamics_eval_dataset = load_dataset(
                    self.configs["checkpointing"].learning_dynamics.eval_data,
                    split="val",
                )
            else:
                self.learning_dynamics_eval_dataset = None

    def train(self) -> None:
        """Execute the main training workflow.

        This method orchestrates the complete training process by:
        1. Creating an initial checkpoint to save the starting state and evaluate the model as a
            baseline
        2. Running the main training loop via `_training_loop`
        3. Handling final checkpointing and evaluation

        The training progress is tracked through checkpoints and evaluations
        at intervals specified in the configuration.
        """

        ########################################################
        #
        # Initial Checkpointing and Evaluation
        #
        ########################################################

        # Save Initial Checkpoint -- If the checkpoint already exists, this performs a no-op
        save_checkpoint(
            configs=self.configs,
            checkpoint_step=self.initial_batch_step,
            fabric=self.fabric,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            tokenizer=self.tokenizer,
        )

        # Save Initial Evaluation Results
        if self.should_evaluate:
            if self.initial_batch_step == 0:
                evaluation_results = run_evaluation(
                    evaluation_config=self.configs["evaluation"],
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                )
                self._log_evaluation_results(
                    evaluation_results, self.initial_batch_step
                )
                save_evaluation_results(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    evaluation_results=evaluation_results,
                    checkpoint_step=self.initial_batch_step,
                )
            else:
                # NOTE: If the run crashed while evaluating, we need to restart the evaluation
                eval_results_path = os.path.join(
                    self.configs["checkpointing"].evaluation.eval_results_dir,
                    f"step_{self.initial_batch_step}.json",
                )
                if not os.path.exists(eval_results_path):
                    evaluation_results = run_evaluation(
                        evaluation_config=self.configs["evaluation"],
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                    )
                    self._log_evaluation_results(
                        evaluation_results, self.initial_batch_step
                    )
                    save_evaluation_results(
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        evaluation_results=evaluation_results,
                        checkpoint_step=self.initial_batch_step,
                    )

        ########################################################
        #
        # Main Training Loop (see `_training_loop` for details)
        #
        ########################################################

        if self.initial_batch_step < self.configs["training"].max_steps:
            self._log_training_configuration()
            final_step = self._training_loop()
        else:
            final_step = self.initial_batch_step

        ########################################################
        #
        # Final Checkpointing and Evaluation
        #
        ########################################################

        # Save Learning Dynamics States
        if self.should_compute_learning_dynamics:
            if self.learning_dynamics_eval_dataset is not None:
                self.log(f"Step {final_step} -- 📈 Saving Learning Dynamics")
                learning_dynamics_val_states = compute_learning_dynamics_states(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                    dataset=self.learning_dynamics_eval_dataset,
                    compute_gradients=True,
                )
                save_learning_dynamics_states(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    learning_dynamics_states=learning_dynamics_val_states,
                    checkpoint_step=final_step,
                    prefix="val",
                )

        # Handle checkpointing and final evaluation
        if final_step % self.configs["checkpointing"].save_every_n_steps != 0:
            self.log(f"Step {final_step} -- 💾 Saving Final Checkpoint")
            save_checkpoint(
                configs=self.configs,
                checkpoint_step=final_step,
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                tokenizer=self.tokenizer,
            )

            # Final evaluation
            if self.should_evaluate:
                evaluation_results = run_evaluation(
                    evaluation_config=self.configs["evaluation"],
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                )
                self._log_evaluation_results(evaluation_results, final_step)
                save_evaluation_results(
                    checkpointing_config=self.configs["checkpointing"],
                    checkpoint_step=final_step,
                    fabric=self.fabric,
                    evaluation_results=evaluation_results,
                )

        self.log(f"🎉 Training complete! Final step: {final_step}")

        if final_step < self.configs["training"].max_steps:
            self.log(
                f"\t Note: Training stopped before max steps ({self.configs['training'].max_steps})",
                level=logging.WARNING,
            )

        # Cleanup distributed training
        self.fabric.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            del self.train_dataloader  # NOTE: shutting down worker nodes

        self.fabric.barrier()

    def _training_loop(self) -> int:
        """Execute the main training loop.

        This method orchestrates the core training loop and includes the following features:
            - Gradient accumulation
            - Gradient clipping
            - Periodic model evaluation and checkpointing
            - Learning Dynamics Checkpointing
            - Learning rate scheduling
            - Logging of training metrics including loss and learning rate
            - Handling of infinite/NaN losses

        Returns:
            int: The final step count reached during training.
                NOTE: A complete training run should match the configured max_steps.
        """
        # Setup training loop variables
        batch_step = self.initial_batch_step

        # NOTE: these are used to compute the average loss over a training interval.
        # This is more accurate than using the loss at the end of the interval.
        interval_loss = torch.tensor(0.0, device=self.fabric.device)
        interval_steps = torch.tensor(0, device=self.fabric.device)
        interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

        if self.configs["model"].rank_normalization_strategy == "orthogonality_loss":
            interval_orthogonality_loss = torch.tensor(0.0, device=self.fabric.device)
        elif self.configs["model"].rank_normalization_strategy == "frobenius_loss":
            interval_frobenius_loss = torch.tensor(0.0, device=self.fabric.device)

        if self.should_compute_learning_dynamics:
            # NOTE: we basically re-construct the full batch here so that we can compute learning dynamics
            training_batch = {"input_ids": []}

        # NOTE: determine what sub-batch we should start from
        initial_sub_batch_step = (
            batch_step
            * self.configs["training"].optimization.gradient_accumulation_steps
        )

        ###############################################################
        #
        # Core loop starts here
        # NOTE: the ratio between sub_batch_step and batch_step
        # is the configured number of gradient_accumulation_steps
        # i.e. with 32 configured gradient accumulation steps,
        # there are 32 sub_batch_steps for each batch_step
        #
        ###############################################################

        for sub_batch_step, sub_batch in enumerate(
            self.train_iterator, start=initial_sub_batch_step
        ):
            # NOTE: We want to store the entire training batch whenever we are computing learning dynamics
            # and we are at a checkpointing step.
            should_store_training_batch = self.should_compute_learning_dynamics and (
                batch_step % self.configs["checkpointing"].save_every_n_steps == 0
            )

            ########################################################
            #
            # Forward Pass
            #
            ########################################################

            _input_ids = torch.tensor(sub_batch["input_ids"], device=self.fabric.device)
            input_ids = _input_ids[:, :-1]
            labels = _input_ids[:, 1:]

            if should_store_training_batch:
                gathered_input_ids = self.fabric.all_gather(_input_ids)

                # NOTE: On multi-GPU, we need to reshape the input_ids to be a 2D tensor; on
                # a single GPU, the input_ids are already a 2D tensor.
                if self.fabric.world_size > 1:
                    gathered_input_ids = gathered_input_ids.reshape(
                        -1, *gathered_input_ids.shape[2:]
                    )

                training_batch["input_ids"].extend(gathered_input_ids.tolist())

            # Forward pass
            model_output, _ = self.model(input_ids)
            model_output = model_output.transpose(1, 2)

            ########################################################
            #
            # Gradient accumulation
            #
            ########################################################

            should_accumulate_gradients = (sub_batch_step + 1) % self.configs[
                "training"
            ].optimization.gradient_accumulation_steps != 0

            with self.fabric.no_backward_sync(
                self.model, enabled=should_accumulate_gradients
            ):
                loss = F.cross_entropy(model_output, labels)

                # Compute special losses (orthogonality, frobenius, normalization)
                if (
                    self.configs["model"].rank_normalization_strategy
                    == "orthogonality_loss"
                ):
                    orthogonality_loss = (
                        self.model.get_orthogonality_loss()
                        * self.configs["model"].rank_normalization_loss_weight
                    )

                    loss += orthogonality_loss.to(loss.device)
                elif (
                    self.configs["model"].rank_normalization_strategy
                    == "frobenius_loss"
                ):
                    frobenius_loss = (
                        self.model.get_frobenius_loss()
                        * self.configs["model"].rank_normalization_loss_weight
                    )
                    loss += frobenius_loss.to(loss.device)

                self.fabric.backward(
                    loss
                    / self.configs["training"].optimization.gradient_accumulation_steps,
                    model=self.model,
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    interval_inf_or_nan_count += 1
                else:
                    interval_loss += loss.item()
                    interval_steps += 1

                    if (
                        self.configs["model"].rank_normalization_strategy
                        == "orthogonality_loss"
                    ):
                        interval_orthogonality_loss += orthogonality_loss.item()
                    elif (
                        self.configs["model"].rank_normalization_strategy
                        == "frobenius_loss"
                    ):
                        interval_frobenius_loss += frobenius_loss.item()

            # NOTE: if we are not accumulating gradients, we should skip the logging and optimization steps
            if should_accumulate_gradients:
                continue

            ########################################################
            #
            # Logging
            #
            ########################################################

            if batch_step % self.configs["monitoring"].logging.log_every_n_steps == 0:
                self._log_training_metrics(
                    interval_loss=interval_loss,
                    interval_steps=interval_steps,
                    interval_inf_or_nan_count=interval_inf_or_nan_count,
                    batch_step=batch_step,
                    **{
                        "interval_orthogonality_loss": interval_orthogonality_loss
                        if self.configs["model"].rank_normalization_strategy
                        == "orthogonality_loss"
                        else None,
                        "interval_frobenius_loss": interval_frobenius_loss
                        if self.configs["model"].rank_normalization_strategy
                        == "frobenius_loss"
                        else None,
                    },
                )
                interval_loss = torch.tensor(0.0, device=self.fabric.device)
                interval_steps = torch.tensor(0, device=self.fabric.device)
                interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

                if (
                    self.configs["model"].rank_normalization_strategy
                    == "orthogonality_loss"
                ):
                    interval_orthogonality_loss = torch.tensor(
                        0.0, device=self.fabric.device
                    )
                elif (
                    self.configs["model"].rank_normalization_strategy
                    == "frobenius_loss"
                ):
                    interval_frobenius_loss = torch.tensor(
                        0.0, device=self.fabric.device
                    )

            ########################################################
            #
            # Learning Dynamics Checkpointing
            #
            ########################################################

            if batch_step % self.configs["checkpointing"].save_every_n_steps == 0:
                if self.should_compute_learning_dynamics:
                    self.log(f"Step {batch_step} -- 📈 Saving Learning Dynamics")

                    # Training Batch Learning Dynamics
                    training_batch_dataset = Dataset.from_dict(training_batch)

                    learning_dynamics_train_states = compute_learning_dynamics_states(
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                        dataset=training_batch_dataset,
                        compute_gradients=True,
                    )

                    save_learning_dynamics_states(
                        checkpointing_config=self.configs["checkpointing"],
                        checkpoint_step=batch_step,
                        prefix="train",
                        fabric=self.fabric,
                        learning_dynamics_states=learning_dynamics_train_states,
                        learning_dynamics_dataset=training_batch_dataset,
                        tokenizer=self.tokenizer,
                    )
                    training_batch = {
                        "input_ids": []
                    }  # Resetting training_batch for next training batch

                    # Validation Data Learning Dynamics
                    if self.learning_dynamics_eval_dataset is not None:
                        learning_dynamics_val_states = compute_learning_dynamics_states(
                            checkpointing_config=self.configs["checkpointing"],
                            fabric=self.fabric,
                            model=self.model,
                            dataset=self.learning_dynamics_eval_dataset,
                            compute_gradients=True,
                        )
                        save_learning_dynamics_states(
                            checkpointing_config=self.configs["checkpointing"],
                            checkpoint_step=batch_step,
                            prefix="val",
                            fabric=self.fabric,
                            learning_dynamics_states=learning_dynamics_val_states,
                        )

            ########################################################
            #
            # Optimization step
            #
            ########################################################

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            batch_step += 1

            ########################################################
            #
            # Training Checkpointing and evaluation
            #
            ########################################################

            if batch_step % self.configs["checkpointing"].save_every_n_steps == 0:
                self.log(f"Step {batch_step} -- 💾 Saving Checkpoint")
                save_checkpoint(
                    configs=self.configs,
                    checkpoint_step=batch_step,
                    fabric=self.fabric,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    tokenizer=self.tokenizer,
                )

                if self.should_evaluate:
                    evaluation_results = run_evaluation(
                        evaluation_config=self.configs["evaluation"],
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                    )
                    if evaluation_results is not None:
                        self._log_evaluation_results(evaluation_results, batch_step)
                        save_evaluation_results(
                            checkpointing_config=self.configs["checkpointing"],
                            fabric=self.fabric,
                            evaluation_results=evaluation_results,
                            checkpoint_step=batch_step,
                        )

            # Break if we've reached training steps
            if batch_step >= self.configs["training"].max_steps:
                break

        return batch_step

    ########################################################
    #
    # Trainer Logging Functinalities
    #
    ########################################################

    def _log_training_metrics(
        self,
        interval_loss: torch.Tensor,
        interval_steps: torch.Tensor,
        interval_inf_or_nan_count: torch.Tensor,
        batch_step: int,
        **additional_losses,
    ):
        """
        Gathers together the training metrics computed across all processes in distributed training
        and logs them in a tree-style format.
        """
        gathered_interval_loss = self.fabric.all_reduce(
            interval_loss, reduce_op="sum"
        ).item()
        gathered_interval_inf_or_nan_count = self.fabric.all_reduce(
            interval_inf_or_nan_count, reduce_op="sum"
        ).item()
        gathered_interval_steps = self.fabric.all_reduce(
            interval_steps, reduce_op="sum"
        ).item()

        gathered_avg_additional_losses = {}
        for loss_name, loss_value in additional_losses.items():
            if loss_value is not None:
                gathered_avg_additional_losses[loss_name] = (
                    self.fabric.all_reduce(loss_value, reduce_op="sum").item()
                    / gathered_interval_steps
                )

        avg_loss = (
            gathered_interval_loss / gathered_interval_steps
            if gathered_interval_steps > 0
            else float("inf")
        )

        self.fabric.log("train/loss", avg_loss, step=batch_step)
        self.fabric.log(
            "trainer/inf_or_nan_count",
            gathered_interval_inf_or_nan_count,
            step=batch_step,
        )
        self.fabric.log(
            "trainer/learning_rate",
            self.lr_scheduler.get_last_lr()[0],
            step=batch_step,
        )

        ce_loss = avg_loss

        for loss_name, loss_value in gathered_avg_additional_losses.items():
            self.fabric.log(
                f"train/{loss_name.split('interval_')[1]}", loss_value, step=batch_step
            )
            ce_loss -= loss_value

        self.fabric.log("train/ce_loss", ce_loss, step=batch_step)

        # Log to console in tree format
        self.log(f"Step {batch_step} -- 🔄 Training Metrics")
        self.log(f"├── Total Loss: {avg_loss:.4f}")
        for loss_name, loss_value in gathered_avg_additional_losses.items():
            # Making the loss name more readable
            _loss_name_split = loss_name.split("interval_")[1].split("_")
            _loss_name = (
                _loss_name_split[0].capitalize()
                + " "
                + _loss_name_split[1].capitalize()
            )
            self.log(f"├── {_loss_name}: {loss_value:.4f}")
        self.log(f"├── CE Loss: {ce_loss:.4f}")
        self.log(f"├── Learning Rate: {self.lr_scheduler.get_last_lr()[0]:.2e}")
        self.log(f"└── Inf/NaN count: {gathered_interval_inf_or_nan_count}")

    def _log_evaluation_results(
        self, evaluation_results: Dict[str, Any], batch_step: int
    ):
        """Log model evaluation metrics to experiment tracking system and console."""
        if self.fabric.global_rank == 0:
            self.log(f"Step {batch_step} -- 📊 Evaluation Results")
            for i, (metric, result) in enumerate(evaluation_results.items()):
                prefix = "└──" if i == len(evaluation_results) - 1 else "├──"
                self.log(f"{prefix} {metric}: {result}")
                self.fabric.log(f"eval/{metric}", result, step=batch_step)

    def _log_training_configuration(self):
        """
        Log training configuration details as well as runtime information about the hardware,
        software, and batch settings.

        This function is called at the beginning of the training loop to provide a summary of the
        training configuration.
        """

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        global_batch_size = self.configs["data"].dataloader.batch_size
        per_device_batch_size = self.train_dataloader.batch_size
        gradient_accumulation_steps = self.configs[
            "training"
        ].optimization.gradient_accumulation_steps

        device_type = ""
        fabric_device = str(self.fabric.device)
        if torch.cuda.is_available() and "cuda" in fabric_device:
            device_type = torch.cuda.get_device_name(self.fabric.device)
        elif torch.backends.mps.is_available() and "mps" in fabric_device:
            device_type = "MPS (Apple Silicon)"
        else:
            device_type = "CPU"

        training_config_path = os.path.join(
            self.configs["checkpointing"].runs_dir,
            self.configs["checkpointing"].run_name,
            "training_config.yaml",
        )
        if os.path.exists(training_config_path):
            self.log("=" * 50)
            self.log("✨ Training Configuration")
            self.log("=" * 50)
            training_config = yaml.safe_load(open(training_config_path, "r"))
            pretty_print_yaml_config(self.logger, training_config)

        self.log("=" * 50)
        self.log("⛭ Runtime Summary:")
        self.log("=" * 50)
        self.log(f"Starting from step: {self.initial_batch_step}")

        self.log("Model Setup:")
        self.log(f"└─ Total Parameters: {total_params:,}")
        self.log(f"└─ Trainable Parameters: {trainable_params:,}")

        self.log("Distributed Setup:")
        self.log(f"└─ Number of Devices: {self.fabric.world_size}")
        self.log(f"└─ Device Type: {device_type}")
        self.log(
            f"└─ Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            if torch.cuda.is_available()
            else f"└─ Available Memory: {psutil.virtual_memory().total / 1e9:.2f} GB"
        )

        self.log("Software Setup:")
        self.log(f"└─ Python Version: {platform.python_version()}")
        self.log(f"└─ PyTorch Version: {torch.__version__}")
        self.log(
            f"└─ CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}"
        )
        self.log(f"└─ Operating System: {platform.system()} {platform.release()}")

        self.log("Batch Size Configuration:")
        self.log(f"└─ Global Batch Size: {global_batch_size}")
        self.log(f"└─ Per Device Batch Size: {per_device_batch_size}")
        self.log(f"└─ Gradient Accumulation Steps: {gradient_accumulation_steps}")
        self.log("=" * 50)

    @rank_zero_only
    def log(self, msg: str, level: int = logging.INFO) -> None:
        """Log messages only from rank zero process."""
        self.logger.log(level, msg)
