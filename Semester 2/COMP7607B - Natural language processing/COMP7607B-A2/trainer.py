import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import DPODataset, PretrainDataset, SFTDataset
from evaluator import CometEvaluator, Evaluator, PerplexityEvaluator
from model.config import LMConfig
from model.lora import apply_lora
from model.model import MiniMindLM

DEFAULT_VAL_SIZE = 1000


def logits_to_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Convert logits to log probabilities for the given labels."""
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs: torch.Tensor, probs: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Calculate DPO loss between reference and policy model probabilities."""
    # Average probabilities across sequence length
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # Split batch into chosen and rejected
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[: batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2 :]
    chosen_probs = probs[: batch_size // 2]
    reject_probs = probs[batch_size // 2 :]

    # Calculate loss
    # Write Your Code Here
    logits_diff = chosen_probs - reject_probs
    ref_logits_diff = chosen_ref_probs - reject_ref_probs
    loss = -F.logsigmoid(beta * (logits_diff - ref_logits_diff))
    
    return loss.mean()


class TrainerBase:
    prev_category: str
    category: str
    dataset_cls: type[Dataset]
    evaluator_cls: type[Evaluator]

    def __init__(self, args):
        self.args = args
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.start_epoch = 0
        self.setup_environment()
        self.setup_model()
        self.setup_dataloader()
        self.setup_evaluator()
        self.setup_training()
        self.load_checkpoint()

    def setup_environment(self):
        # Set up DDP if enabled
        self.ddp_local_rank, self.device = 0, "cuda:0"
        if self.ddp:
            self.init_distributed_mode()
            self.args.device = torch.device(self.device)

        # Create output directories
        os.makedirs(self.args.out_dir, exist_ok=True)

        # Set up context for mixed precision training
        device_type = "cuda" if "cuda" in self.args.device else "cpu"
        self.ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

        # Set random seed
        torch.manual_seed(1337)

        # Initialize wandb if enabled
        self.wandb = None
        if self.args.use_wandb and (not self.ddp or self.ddp_local_rank == 0):
            import wandb

            self.wandb = wandb
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name)

    def init_distributed_mode(self):
        dist.init_process_group(backend="nccl")
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)

    def log(self, content):
        """Log messages only from rank 0 in DDP mode"""
        if not self.ddp or self.ddp_local_rank == 0:
            print(content)

    def setup_model(self):
        # Initialize model and tokenizer
        self.lm_config = LMConfig(
            dim=self.args.dim,
            n_layers=self.args.n_layers,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer")

        # Build model
        self.model = MiniMindLM(self.lm_config).to(self.args.device)

        # Get model parameters
        self.model_parameters = self.model.parameters()

    def setup_dataloader(self):
        # Set up dataset and dataloader
        # Initialize dataset
        ds = self.dataset_cls(
            self.args.data_path,
            self.tokenizer,
            max_length=self.args.max_seq_len,
        )

        # Get train and validation sets
        train_ds = ds.train_ds
        val_ds = ds.val_ds

        # Initialize train dataloader
        train_sampler = DistributedSampler(train_ds) if self.ddp else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.args.num_workers,
            sampler=train_sampler,
        )
        self.iter_per_epoch = len(self.train_loader)

        # Initialize validation dataloader
        val_sampler = DistributedSampler(val_ds) if self.ddp else None
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.args.num_workers,
            sampler=val_sampler,
        )
        self.log(f"Train dataset size: {len(train_ds)}")
        self.log(f"Validation dataset size: {len(val_ds)}")

    def setup_evaluator(self):
        self.evaluator = self.evaluator_cls(self)

    def setup_training(self):
        # Initialize optimizer, loss function, and scaler
        self.optimizer = optim.AdamW(self.model_parameters, lr=self.args.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype in ["float16", "bfloat16"]))
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Set up DDP model if needed
        if self.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])

    def get_lr(self, current_step, total_steps):
        """Calculate learning rate with cosine decay schedule"""
        lr = self.args.learning_rate
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

    def train_epoch(self, epoch):
        """Train for one epoch"""
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            X = X.to(self.args.device)
            Y = Y.to(self.args.device)
            loss_mask = loss_mask.to(self.args.device)

            # Update learning rate
            current_step = epoch * self.iter_per_epoch + step
            total_steps = self.args.epochs * self.iter_per_epoch
            lr = self.get_lr(current_step, total_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass with mixed precision
            with self.ctx:
                res = self.model(X)
                loss = self.loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss = loss / self.args.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights if accumulation steps reached
            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_parameters, self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Logging
            if step % self.args.log_interval == 0:
                self.log_progress(epoch, step, loss, start_time)

        # Save checkpoint at epoch end
        if not self.ddp or self.ddp_local_rank == 0:
            self.save_checkpoint(epoch=epoch)

    def log_progress(self, epoch, step, loss, start_time):
        """Log training progress"""
        spend_time = time.time() - start_time
        estimated_epoch_time = spend_time / (step + 1) * self.iter_per_epoch // 60 - spend_time // 60

        self.log(
            f"Epoch:[{epoch + 1}/{self.args.epochs}]({step}/{self.iter_per_epoch}) "
            f"loss:{loss.item():.3f} "
            f"lr:{self.optimizer.param_groups[-1]['lr']:.12f} "
            f"epoch_Time:{estimated_epoch_time}min:"
        )

        if (self.wandb is not None) and (not self.ddp or self.ddp_local_rank == 0):
            self.wandb.log(
                {
                    "loss": loss,
                    "lr": self.optimizer.param_groups[-1]["lr"],
                    "epoch_Time": estimated_epoch_time,
                }
            )

    def run(self):
        """Run the full training process"""
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_epoch(epoch)

    def eval(self):
        """Evaluate the model"""
        self.evaluator.eval()

    def get_predictions(self, messages_lst):
        """
        Get predictions from the model with batch support
        """
        self.model.eval()
        predictions: list[str] = []

        with torch.no_grad():
            for messages in tqdm(messages_lst, desc="Generating predictions"):
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                x = torch.tensor(self.tokenizer(prompt)["input_ids"], device=self.args.device).unsqueeze(0)
                outputs = self.model.generate(
                    x,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.args.max_seq_len,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                predictions.append(
                    self.tokenizer.decode(outputs.squeeze()[x.shape[1] :].tolist(), skip_special_tokens=True)
                )

        return predictions

    def load_checkpoint(self):
        """Load model checkpoint for continuing training"""
        prev_ckp = f"{self.args.out_dir}/{self.prev_category}_{self.lm_config.dim}.pth"
        curr_ckp = f"{self.args.out_dir}/{self.category}_{self.lm_config.dim}.pth"

        if os.path.isfile(curr_ckp):
            self.log(f"Loading checkpoint from {curr_ckp}")
            self._load_checkpoint_from_continue_training(curr_ckp)
            self.log(f"Checkpoint {curr_ckp} loaded successfully")
        else:
            self.log(f"Loading checkpoint from {prev_ckp}")
            self._load_checkpoint_from_prev_stage(prev_ckp)
            self.log(f"Checkpoint {prev_ckp} loaded successfully")

    def _load_checkpoint_from_continue_training(self, ckp):
        checkpoint = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint["epoch"] + 1

    def _load_checkpoint_from_prev_stage(self, ckp):
        checkpoint = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = 0

    def save_checkpoint(self, epoch):
        """Save model checkpoint with training state"""
        self.model.eval()
        checkpoint = {
            "model_state_dict": self.model.module.state_dict() if self.ddp else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": epoch,
        }

        checkpoint_path = f"{self.args.out_dir}/{self.category}_{self.lm_config.dim}.pth"
        torch.save(checkpoint, checkpoint_path)

        self.log(f"Checkpoint saved to {checkpoint_path}")
        self.model.train()


class PreTrainer(TrainerBase):
    prev_category = ""
    category: str = "pretrain"
    dataset_cls: type[Dataset] = PretrainDataset
    evaluator_cls: type[Evaluator] = PerplexityEvaluator

    def _load_checkpoint_from_prev_stage(self, ckp):
        # No previous stage checkpoint for pretraining
        self.start_epoch = 0

    def get_predictions(self, messages_lst):
        raise NotImplementedError("Pretraining does not support prediction generation.")


class SFTTrainer(TrainerBase):
    prev_category: str = "pretrain"
    category: str = "sft"
    dataset_cls: type[Dataset] = SFTDataset
    evaluator_cls: type[Evaluator] = CometEvaluator


class LoraTrainer(SFTTrainer):
    prev_category: str = "sft"
    category: str = "lora"

    def __init__(self, args):
        super().__init__(args)
        self.freeze_non_lora_parameters()

    def _load_checkpoint_from_prev_stage(self, ckp):
        checkpoint = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.start_epoch = 0

    def freeze_non_lora_parameters(self):
        """Freeze all parameters except LoRA parameters"""
        # Freeze non-LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def setup_model(self):
        super().setup_model()

        # Apply lora
        self.model = apply_lora(self.model, self.args.lora_rank)

        # Collect LoRA parameters for optimization
        self.model_parameters = [param for name, param in self.model.named_parameters() if "lora" in name]

        # Log parameter statistics
        if not self.ddp or self.ddp_local_rank == 0:
            lora_params_count = sum(p.numel() for p in self.model_parameters)
            self.log(f"LoRA parameters: {lora_params_count:,}")


class DPOTrainer(TrainerBase):
    prev_category: str = "sft"
    category: str = "dpo"
    dataset_cls: type[Dataset] = DPODataset
    evaluator_cls: type[Evaluator] = CometEvaluator

    def setup_model(self):
        # Initialize model and tokenizer
        self.lm_config = LMConfig(
            dim=self.args.dim,
            n_layers=self.args.n_layers,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer")

        # Build model
        self.model = MiniMindLM(self.lm_config).to(self.args.device)
        self.ref_model = MiniMindLM(self.lm_config).to(self.args.device)

        # Get model parameters
        self.model_parameters = self.model.parameters()

    def _load_checkpoint_from_prev_stage(self, ckp):
        state_dict = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.ref_model.load_state_dict(state_dict["model_state_dict"])
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.start_epoch = 0

    def _load_checkpoint_from_continue_training(self, ckp):
        checkpoint = torch.load(ckp, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ref_model.load_state_dict(checkpoint["ref_model_state_dict"])
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.start_epoch = checkpoint["epoch"] + 1

    def train_epoch(self, epoch):
        """Train for one epoch"""
        start_time = time.time()
        for step, batch in enumerate(self.train_loader):
            # Prepare inputs
            x_chosen = batch["x_chosen"].to(self.args.device)
            x_rejected = batch["x_rejected"].to(self.args.device)
            y_chosen = batch["y_chosen"].to(self.args.device)
            y_rejected = batch["y_rejected"].to(self.args.device)
            mask_chosen = batch["mask_chosen"].to(self.args.device)
            mask_rejected = batch["mask_rejected"].to(self.args.device)

            # Concatenate chosen and rejected samples
            x = torch.cat([x_chosen, x_rejected], dim=0)
            y = torch.cat([y_chosen, y_rejected], dim=0)
            mask = torch.cat([mask_chosen, mask_rejected], dim=0)

            # Update learning rate
            current_step = epoch * self.iter_per_epoch + step
            total_steps = self.args.epochs * self.iter_per_epoch
            lr = self.get_lr(current_step, total_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Forward pass with mixed precision
            with self.ctx:
                # Get reference model outputs (no gradient)
                with torch.no_grad():
                    ref_outputs = self.ref_model(x)
                    ref_logits = ref_outputs.logits
                ref_probs = logits_to_probs(ref_logits, y)
                ref_probs = ref_probs * mask

                # Get policy model outputs
                outputs = self.model(x)
                logits = outputs.logits
                probs = logits_to_probs(logits, y)
                probs = probs * mask

                # Calculate DPO loss
                loss = dpo_loss(ref_probs, probs, beta=0.1)
                loss = loss / self.args.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights if accumulation steps reached
            if (step + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_parameters, self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Logging
            if step % self.args.log_interval == 0:
                self.log_progress(epoch, step, loss, start_time)

        # Save checkpoint at epoch end
        if not self.ddp or self.ddp_local_rank == 0:
            self.save_checkpoint(epoch=epoch)

    def save_checkpoint(self, epoch):
        """Save model checkpoint with training state"""
        self.model.eval()
        checkpoint = {
            "model_state_dict": self.model.module.state_dict() if self.ddp else self.model.state_dict(),
            "ref_model_state_dict": self.ref_model.module.state_dict() if self.ddp else self.ref_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": epoch,
        }

        checkpoint_path = f"{self.args.out_dir}/{self.category}_{self.lm_config.dim}.pth"
        torch.save(checkpoint, checkpoint_path)

        self.log(f"Checkpoint saved to {checkpoint_path}")
        self.model.train()
