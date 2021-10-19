########## The following part is copied from Transformers' trainer (3.4.0) ########## 

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import torch
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn
import transformers
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_utils import (
    EvalPrediction,
    TrainOutput,
    set_seed,
)

from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.training_args import TrainingArguments
from transformers.utils import logging

from tqdm import tqdm, trange
from .utils import mask_tokens

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)


logger = logging.get_logger(__name__)


def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]
 
    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            unlabeled_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            **kwargs,
    ):
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args
        logger.info(self.args)
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        assert (
                model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` argument."
        self.model_init = model_init
        if model is None and model_init is not None:
            model = self.call_model_init()
        self.model = model.to(args.device) if model is not None else None
        default_collator = default_data_collator
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.prob_list = []
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(callbacks, self.model, self.optimizer, self.lr_scheduler)
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Deprecated arguments
        if "tb_writer" in kwargs:
            warnings.warn(
                "Passing `tb_writer` as a keyword argument is deprecated and won't be possible in a "
                + "future version. Use `TensorBoardCallback(tb_writer=...)` instead and pass it to the `callbacks`"
                + "argument",
                FutureWarning,
            )
            tb_writer = kwargs.pop("tb_writer")
            self.remove_callback(TensorBoardCallback)
            self.add_callback(TensorBoardCallback(tb_writer=tb_writer))
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a "
                + "future version. Use `args.prediction_loss_only` instead. Setting "
                + f"`args.prediction_loss_only={kwargs['prediction_loss_only']}",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available() and isinstance(self.model, PreTrainedModel):
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                        "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                        + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        # Enforce rules on using datasets with no __len__
        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")
        if unlabeled_dataset is not None and not isinstance(unlabeled_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("unlabeled_dataset does not implement __len__, max_steps has to be specified")
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        if is_datasets_available():
            if isinstance(train_dataset, datasets.Dataset):
                self._remove_unused_columns(self.train_dataset, description="training")
            if isinstance(unlabeled_dataset, datasets.Dataset):
                self._remove_unused_columns(self.unlabeled_dataset, description="self-training")
            if isinstance(eval_dataset, datasets.Dataset):
                self._remove_unused_columns(self.eval_dataset, description="evaluation")

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable for total_flos used to count as tensors (for distributed + TPU), will be sent in the
        # state at each call to self.log.
        self._total_flos = None
        if self.args.fp16 and _use_native_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions, end_positions"]
            if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def _get_unlabeled_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.unlabeled_dataset, collections.abc.Sized):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.unlabeled_dataset)
        else:
            return (
                RandomSampler(self.unlabeled_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.unlabeled_dataset)
            )

    def get_unlabeled_dataloader(self) -> DataLoader:
        if self.unlabeled_dataset is None:
            raise ValueError("Trainer: self-training requires a unlabeled_dataset.")
        unlabeled_sampler = self._get_unlabeled_sampler()

        return DataLoader(
            self.unlabeled_dataset,
            batch_size=int(self.args.train_batch_size * self.args.mu),
            sampler=unlabeled_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def set_mu(self):
        self.args.mu = int(len(self.unlabeled_dataset) / len(self.train_dataset))
        assert len(self.unlabeled_dataset) % len(self.train_dataset) == 0

    def compute_st_loss(self, model, unlabeled_inputs):
        unlabeled_inputs_aug = copy.deepcopy(unlabeled_inputs)
        unlabeled_inputs_aug['input_ids'], _ = mask_tokens(inputs=unlabeled_inputs_aug['input_ids'],
                                                           tokenizer=self.tokenizer)

        unlabeled_outputs = model(**unlabeled_inputs)
        unlabeled_logits = unlabeled_outputs[1]
        unlabeled_outputs_aug = model(**unlabeled_inputs_aug)
        unlabeled_logits_aug = unlabeled_outputs_aug[1]

        with torch.no_grad():
            probs = torch.softmax(unlabeled_logits, dim=1)
            self.prob_list.append(probs.mean(0))
            if len(self.prob_list) > 32:
                self.prob_list.pop(0)

            prob_avg = torch.stack(self.prob_list, dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            scores, unlabeled_guess = torch.max(probs, dim=1)
            mask = scores.ge(self.args.threshold).float()
        st_loss_fct = nn.CrossEntropyLoss(reduction='none').cuda()
        st_loss = (st_loss_fct(unlabeled_logits_aug, unlabeled_guess) * mask).mean()
        return st_loss

    def compute_mlm_loss(self, model, unlabeled_inputs):
        unlabeled_inputs_mlm = copy.deepcopy(unlabeled_inputs)
        unlabeled_inputs_mlm['input_ids'], unlabeled_inputs_mlm['labels'] = mask_tokens(inputs=unlabeled_inputs_mlm['input_ids'],
                                                                                        tokenizer=self.tokenizer)
        mlm_outputs = model(**unlabeled_inputs_mlm, mlm=True)
        mlm_loss = mlm_outputs[0]
        return mlm_loss

    def compute_loss(self, model, labeled_inputs, unlabeled_inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labeled_outputs = model(**labeled_inputs)
        labeled_loss = labeled_outputs[0]
        if self.args.use_st_loss:
            st_loss = self.compute_st_loss(model, unlabeled_inputs)
        if self.args.use_mlm_loss:
            mlm_loss = self.compute_mlm_loss(model, unlabeled_inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = labeled_outputs[self.args.past_index]
        # We don't use .loss here since the model may return tuples instead of ModelOutput.

        loss = labeled_loss
        if self.args.use_st_loss:
            loss = loss + self.args.lam1 * st_loss
        if self.args.use_mlm_loss:
            loss = loss + self.args.lam2 * mlm_loss
        return loss

    def training_step(self, model: nn.Module, labeled_inputs: Dict[str, Union[torch.Tensor, Any]], unlabeled_inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        labeled_inputs = self._prepare_inputs(labeled_inputs)
        unlabeled_inputs = self._prepare_inputs(unlabeled_inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss = self.compute_loss(model, labeled_inputs, unlabeled_inputs)
        else:
            loss = self.compute_loss(model, labeled_inputs, unlabeled_inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        self.best_dir = None
        self.prob_list = []
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps 
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
        # Load unlabeled data
        self.set_mu()
        logger.info(self.args.mu)
        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        unlabeled_dataloader = self.get_unlabeled_dataloader()

        # set optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
                unlabeled_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                labeled_parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                labeled_epoch_iterator = tqdm(labeled_parallel_loader, desc="Iteration", disable=not self.is_local_master())
                # add unlabled iterator
                unlabeled_parallel_loader = pl.ParallelLoader(unlabeled_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                unlabeled_epoch_iterator = tqdm(unlabeled_parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                labeled_epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
                unlabeled_epoch_iterator = tqdm(unlabeled_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, (labeled_inputs, unlabeled_inputs) in enumerate(zip(labeled_epoch_iterator, unlabeled_epoch_iterator)):
            # for step, labeled_inputs in enumerate(labeled_epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self.training_step(model, labeled_inputs, unlabeled_inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(labeled_epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(labeled_epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16:
                        norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if transformers.is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(labeled_epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    # ----------------------------------------------------------------------
                    # BEGIN CHANGES.
                    # ----------------------------------------------------------------------

                    metrics = None
                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        output = self.evaluate()
                        metrics = output.metrics
                        objective = self.dev_objective(metrics)
                        if objective > self.objective:
                            logger.info("Best dev result: {}".format(objective))
                            self.objective = objective
                            self.save_model(self.args.output_dir)

                    # ----------------------------------------------------------------------
                    # END CHANGES.
                    # ----------------------------------------------------------------------


                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    labeled_epoch_iterator.close()
                    unlabeled_epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
