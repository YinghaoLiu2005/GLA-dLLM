#!/usr/bin/env python
# coding=utf-8
"""Distillation fine-tuning script for GLADream students.

This script loads a frozen teacher model and a GLADream student model,
computes per-layer hidden-state alignment losses together with the
student language-model loss, and performs supervised distillation.
"""

import glob
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the src directory to Python path for lmflow modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

# Patch Accelerator.unwrap_model for older accelerate versions
try:
    from accelerate import Accelerator

    _original_unwrap_model = Accelerator.unwrap_model

    def _patched_unwrap_model(self, model, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return _original_unwrap_model(self, model, **kwargs)

    Accelerator.unwrap_model = _patched_unwrap_model
except Exception as exc:  # pragma: no cover - best effort patch
    print(f"Warning: Could not patch Accelerator.unwrap_model: {exc}")

from accelerate import dispatch_model  # noqa: E402
from accelerate.utils import infer_auto_device_map  # noqa: E402
from datasets import Dataset as HFDataset  # noqa: E402
from datasets import DatasetDict, load_dataset, load_from_disk  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    get_scheduler,
    set_seed,
)
from transformers.training_args import TrainingArguments  # noqa: E402

from lmflow.args import DatasetArguments, ModelArguments  # noqa: E402

# Import GLADream components
sys.path.append(os.path.join(CURRENT_DIR, "..", "..", "gla_dream"))
try:
    from gla_dream.configuration_gla_dream import GLADreamConfig  # type: ignore  # noqa: E402
    from gla_dream.modeling_gla_dream import GLADreamModel  # type: ignore  # noqa: E402

    GLA_DREAM_AVAILABLE = True
except ImportError:
    GLA_DREAM_AVAILABLE = False
    print("Warning: GLA Dream not available. Distillation script requires gla_dream package.")


TORCH_DTYPE_MAP = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class DistillationArguments(TrainingArguments):
    """Arguments specific to the distillation training run."""

    teacher_model_name_or_path: str = field(
        metadata={"help": "Path or identifier of the frozen teacher model."}
    )
    teacher_revision: str = field(
        default="main", metadata={"help": "Teacher model revision to use."}
    )
    teacher_trust_remote_code: bool = field(
        default=False, metadata={"help": "Allow execution of remote teacher model code."}
    )
    teacher_torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override teacher dtype (auto|float16|bfloat16|float32).",
            "choices": list(TORCH_DTYPE_MAP.keys()),
        },
    )
    teacher_visible_devices: str = field(
        default="0,1",
        metadata={"help": "Comma-separated CUDA device indices hosting the teacher."},
    )
    student_visible_devices: str = field(
        default="2,3",
        metadata={"help": "Comma-separated CUDA device indices hosting the student."},
    )
    teacher_max_memory_per_gpu: Optional[str] = field(
        default=None,
        metadata={"help": "Memory budget per teacher GPU, e.g. '20GiB'."},
    )
    student_max_memory_per_gpu: Optional[str] = field(
        default=None,
        metadata={"help": "Memory budget per student GPU, e.g. '20GiB'."},
    )
    teacher_offload_to_cpu: bool = field(
        default=False,
        metadata={"help": "Permit teacher parameter offload to CPU if needed."},
    )
    student_offload_to_cpu: bool = field(
        default=False,
        metadata={"help": "Permit student parameter offload to CPU if needed."},
    )
    hidden_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for per-layer hidden-state alignment loss."},
    )
    logits_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for KL divergence on logits."},
    )
    ce_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for student cross-entropy loss."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Logit distillation temperature."},
    )
    max_distill_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of layers (including embedding) to match."},
    )
    match_embeddings: bool = field(
        default=True,
        metadata={"help": "Include embedding outputs in hidden-state loss."},
    )
    match_final_only: bool = field(
        default=False,
        metadata={"help": "Only match embedding and final layer if True."},
    )
    normalize_hidden_states: bool = field(
        default=False,
        metadata={"help": "Apply layer norm before computing hidden-state loss."},
    )
    pad_token_id_override: Optional[int] = field(
        default=None,
        metadata={"help": "Override pad token id when deriving attention masks."},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for the training dataloader."},
    )
    dataloader_prefetch_factor: int = field(
        default=2,
        metadata={"help": "Prefetch factor for each dataloader worker."},
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Pin dataloader memory before host-to-device copy."},
    )
    dataset_materialize_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size when materializing preprocessed dataset into RAM."},
    )


def _string_to_devices(devices: str) -> List[int]:
    return [int(x.strip()) for x in devices.split(",") if x.strip()]


def _default_memory_string(device_idx: int) -> str:
    total = torch.cuda.get_device_properties(device_idx).total_memory
    reserve = 2 * 1024**3  # keep ~2GiB headroom
    usable = max(total - reserve, int(1.5 * 1024**3))
    gib = usable / (1024**3)
    return f"{int(gib)}GiB"


def _build_max_memory_map(
    device_ids: Sequence[int], per_gpu_limit: Optional[str], allow_cpu: bool
) -> Dict[object, str]:
    if not torch.cuda.is_available():
        return {"cpu": "400GiB"}

    max_memory: Dict[Sequence[str], str] = {}
    total_gpus = torch.cuda.device_count()
    for gpu_idx in range(total_gpus):
        if gpu_idx in device_ids:
            max_memory[gpu_idx] = per_gpu_limit or _default_memory_string(gpu_idx)
        else:
            max_memory[gpu_idx] = "0GiB"
    max_memory["cpu"] = "400GiB" if allow_cpu else "0GiB"
    return max_memory


def _dispatch_parallel_model(
    model: torch.nn.Module,
    device_ids: Sequence[int],
    per_gpu_limit: Optional[str],
    allow_cpu: bool,
) -> Tuple[torch.nn.Module, Dict[str, int]]:
    if not torch.cuda.is_available() or not device_ids:
        model.to(torch.device("cpu"))
        return model, {"": -1}

    if len(device_ids) == 1:
        target = torch.device(f"cuda:{device_ids[0]}")
        model.to(target)
        return model, {"": device_ids[0]}

    max_memory = _build_max_memory_map(device_ids, per_gpu_limit, allow_cpu)
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=getattr(model, "_no_split_modules", None),
    )
    dispatch_model(model, device_map=device_map)
    return model, device_map


def _freeze_inherited_weights(
    gla_model: torch.nn.Module,
    base_model_path: str,
    trust_remote_code: bool,
) -> None:
    try:
        from transformers import AutoModelForCausalLM

        print(f"[Freeze] Loading base weights from {base_model_path} to identify inherited tensors...")
        reference = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
        )
        reference_state = reference.state_dict()
        del reference
        torch.cuda.empty_cache()

        frozen = 0
        trainable = 0
        for name, param in gla_model.named_parameters():
            if name in reference_state and reference_state[name].shape == param.shape:
                param.requires_grad = False
                frozen += 1
            else:
                param.requires_grad = True
                trainable += 1
        print(f"[Freeze] Frozen {frozen} parameters; {trainable} parameters remain trainable.")
    except Exception as exc:
        print(f"[Freeze] Warning: unable to freeze inherited weights ({exc}).")
        print("[Freeze] Proceeding with all parameters trainable.")


def _load_preprocessed_dataset(
    path: str,
    materialize_batch_size: int,
) -> HFDataset:
    dataset_on_disk = load_from_disk(path)
    if isinstance(dataset_on_disk, DatasetDict):
        if "train" not in dataset_on_disk:
            raise ValueError("Preprocessed DatasetDict must include a 'train' split.")
        dataset_on_disk = dataset_on_disk["train"]

    def _identity(batch):
        return batch

    print("[Data] Materializing preprocessed dataset into RAM (keep_in_memory=True)...")
    materialized = dataset_on_disk.map(
        _identity,
        batched=True,
        batch_size=materialize_batch_size,
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Materializing preprocessed dataset",
    )
    materialized.set_format(type="torch")
    return materialized


def _collect_jsonl_files(dataset_path: str) -> List[str]:
    if os.path.isfile(dataset_path):
        return [dataset_path]

    patterns = ("*.jsonl", "*.json")
    files: List[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(os.path.join(dataset_path, pattern))))
    return files


def _prepare_raw_dataset(
    tokenizer,
    data_args: DatasetArguments,
    model_max_length: int,
) -> HFDataset:
    if data_args.dataset_path is None:
        raise ValueError("dataset_path must be provided when no preprocessed dataset is supplied.")

    jsonl_files = _collect_jsonl_files(data_args.dataset_path)
    if not jsonl_files:
        raise FileNotFoundError(f"No JSON/JSONL files found at {data_args.dataset_path}.")

    max_files = getattr(data_args, "parquet_max_files", None)
    if max_files:
        jsonl_files = jsonl_files[: max_files]

    data_files: Dict[str, object]
    data_files = {"train": jsonl_files if len(jsonl_files) > 1 else jsonl_files[0]}
    print(f"[Data] Loading {len(jsonl_files)} json/jsonl files from {data_args.dataset_path}...")

    raw_dataset = load_dataset("json", data_files=data_files, split="train")
    text_column = "text"

    if data_args.block_size is None:
        if model_max_length > 1024:
            print(
                "[Data] tokenizer.model_max_length is large; default block_size=1024."
                " Override with --block_size to customize."
            )
        block_size = min(1024, model_max_length)
    else:
        if data_args.block_size > model_max_length:
            print(
                f"[Data] Requested block_size {data_args.block_size} exceeds model_max_length {model_max_length}."
                f" Using {model_max_length} instead."
            )
            block_size = model_max_length
        else:
            block_size = data_args.block_size

    disable_group_texts = getattr(data_args, "disable_group_texts", False)
    num_proc = getattr(data_args, "preprocessing_num_workers", None)

    def tokenize_function(examples):
        if disable_group_texts:
            return tokenizer(
                examples[text_column],
                truncation=True,
                max_length=block_size,
                padding=False,
                return_special_tokens_mask=True,
            )
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=model_max_length,
            return_special_tokens_mask=True,
        )

    tokenized_ds = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names,
        desc="[Data] Tokenizing dataset",
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if disable_group_texts:
        print("[Data] disable_group_texts=True: applying per-sample blocking/padding...")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padding_side = tokenizer.padding_side

        def apply_blocking(examples):
            num_examples = len(examples["input_ids"])
            max_length = min(block_size, model_max_length)
            attention_exists = "attention_mask" in examples
            if attention_exists:
                examples["attention_mask"] = [mask.copy() for mask in examples["attention_mask"]]
            else:
                examples["attention_mask"] = [[1] * len(ids) for ids in examples["input_ids"]]
            if "special_tokens_mask" in examples:
                examples["special_tokens_mask"] = [mask.copy() for mask in examples["special_tokens_mask"]]
            for idx in range(num_examples):
                length = len(examples["input_ids"][idx])
                pad_length = max_length - length
                if pad_length < 0:
                    examples["input_ids"][idx] = examples["input_ids"][idx][:max_length]
                    examples["attention_mask"][idx] = examples["attention_mask"][idx][:max_length]
                    if "special_tokens_mask" in examples:
                        examples["special_tokens_mask"][idx] = examples["special_tokens_mask"][idx][:max_length]
                    continue
                if pad_length == 0:
                    continue
                if padding_side == "right":
                    examples["input_ids"][idx].extend([pad_token_id] * pad_length)
                    examples["attention_mask"][idx].extend([0] * pad_length)
                    if "special_tokens_mask" in examples:
                        examples["special_tokens_mask"][idx].extend([1] * pad_length)
                else:
                    examples["input_ids"][idx] = [pad_token_id] * pad_length + examples["input_ids"][idx]
                    examples["attention_mask"][idx] = [0] * pad_length + examples["attention_mask"][idx]
                    if "special_tokens_mask" in examples:
                        examples["special_tokens_mask"][idx] = [1] * pad_length + examples["special_tokens_mask"][idx]
            labels = []
            for idx in range(num_examples):
                label_row = examples["input_ids"][idx].copy()
                labels.append([
                    token if examples["attention_mask"][idx][pos] == 1 else -100
                    for pos, token in enumerate(label_row)
                ])
            examples["labels"] = labels
            return examples

        lm_dataset = tokenized_ds.map(
            apply_blocking,
            batched=True,
            batch_size=1,
            num_proc=num_proc,
            desc="[Data] Blocking samples",
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        print("[Data] Grouping tokenized streams into fixed-size blocks...")
        columns_to_keep = [col for col in ["input_ids", "attention_mask", "special_tokens_mask"] if col in tokenized_ds.column_names]
        if columns_to_keep != tokenized_ds.column_names:
            tokenized_ds = tokenized_ds.remove_columns([c for c in tokenized_ds.column_names if c not in columns_to_keep])

        def group_texts(examples):
            concatenated = {k: list(chain(*examples[k])) for k in examples if k in columns_to_keep}
            total_length = len(concatenated["input_ids"])
            total_length = (total_length // block_size) * block_size
            if total_length == 0:
                return {k: [] for k in columns_to_keep + ["labels"]}
            result = {}
            for k in columns_to_keep:
                result[k] = [
                    concatenated[k][i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
            result["labels"] = [ids.copy() for ids in result["input_ids"]]
            return result

        lm_dataset = tokenized_ds.map(
            group_texts,
            batched=True,
            batch_size=getattr(data_args, "group_texts_batch_size", 1000),
            num_proc=num_proc,
            desc="[Data] Creating LM blocks",
            load_from_cache_file=not data_args.overwrite_cache,
        )

    lm_dataset.set_format(type="torch")
    print(f"[Data] Final processed dataset contains {len(lm_dataset)} samples.")
    return lm_dataset

def prepare_dataset(
    tokenizer,
    data_args: DatasetArguments,
    materialize_batch_size: int,
    model_max_length: int,
) -> HFDataset:
    if data_args.preprocessed_dataset_path and os.path.exists(data_args.preprocessed_dataset_path):
        return _load_preprocessed_dataset(data_args.preprocessed_dataset_path, materialize_batch_size)
    return _prepare_raw_dataset(tokenizer, data_args, model_max_length)

def _select_layer_indices(
    total_layers: int,
    max_layers: Optional[int],
    match_final_only: bool,
) -> List[int]:
    indices = list(range(total_layers))
    if match_final_only and total_layers > 1:
        return [0, total_layers - 1]
    if max_layers is None or max_layers >= total_layers:
        return indices
    if max_layers <= 0:
        return []
    step = max((total_layers - 1) / (max_layers - 1), 1.0)
    sampled = sorted({int(round(i * step)) for i in range(max_layers)})
    if sampled[-1] != total_layers - 1:
        sampled[-1] = total_layers - 1
    return sampled


def hidden_state_alignment_loss(
    student_hidden: Sequence[torch.Tensor],
    teacher_hidden: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    layer_indices: Iterable[int],
    match_embeddings: bool,
    normalize: bool,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    mask = attention_mask.unsqueeze(-1).to(student_hidden[0].dtype)
    mask_tokens = mask.sum().clamp_min(1.0)
    for idx in layer_indices:
        if idx == 0 and not match_embeddings:
            continue
        s = student_hidden[idx]
        t = teacher_hidden[idx].to(s.device).to(s.dtype)
        if normalize:
            s = F.layer_norm(s, s.shape[-1:])
            t = F.layer_norm(t, t.shape[-1:])
        diff = (s - t) * mask
        layer_loss = diff.pow(2).sum() / (mask_tokens * s.shape[-1])
        losses.append(layer_loss)
    if not losses:
        return student_hidden[0].new_zeros(())
    return torch.stack(losses).mean()


def logits_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(kl.dtype)
        kl = kl * mask
        denom = mask.sum().clamp_min(1.0) * kl.shape[-1]
        loss = kl.sum() / denom
    else:
        loss = kl.mean()
    return loss * (temperature ** 2)


def _resolve_pad_token_id(
    tokenizer,
    override: Optional[int],
) -> int:
    if override is not None:
        return override
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Cannot determine pad token id. Provide --pad_token_id_override.")


def _maybe_enable_gradient_checkpointing(model: torch.nn.Module, enable: bool) -> None:
    if enable and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def _gather_trainable_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _primary_device_from_ids(device_ids: Sequence[int]) -> torch.device:
    if not torch.cuda.is_available() or not device_ids:
        return torch.device("cpu")
    return torch.device(f"cuda:{device_ids[0]}")


def save_checkpoint(
    student_model: torch.nn.Module,
    tokenizer,
    output_dir: str,
    step: int,
    save_total_limit: Optional[int],
    use_safetensors: bool,
) -> None:
    checkpoint_dir = os.path.join(output_dir, f"step-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[Checkpoint] Saving student model to {checkpoint_dir}")
    student_model.save_pretrained(checkpoint_dir, safe_serialization=use_safetensors)
    tokenizer.save_pretrained(checkpoint_dir)
    if save_total_limit is not None:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("step-")]
        checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[-1]))
        while len(checkpoints) > save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(os.path.join(output_dir, oldest), ignore_errors=True)


def train():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, DistillationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, distill_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, distill_args = parser.parse_args_into_dataclasses()

    if not GLA_DREAM_AVAILABLE:
        raise RuntimeError("GLADream package is required for distillation training.")

    os.makedirs(distill_args.output_dir, exist_ok=True)
    set_seed(distill_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    gla_config = GLADreamConfig(
        vocab_size=getattr(base_config, "vocab_size", tokenizer.vocab_size),
        hidden_size=getattr(base_config, "hidden_size", 1024),
        num_hidden_layers=getattr(base_config, "num_hidden_layers", 24),
        num_attention_heads=getattr(base_config, "num_attention_heads", 16),
        intermediate_size=getattr(base_config, "intermediate_size", 2816),
        max_position_embeddings=getattr(base_config, "max_position_embeddings", 32768),
        bd_size=getattr(data_args, "bd_size", getattr(base_config, "bd_size", 32)),
    )

    print("[Model] Loading GLADream student model...")
    student_model = GLADreamModel.from_fast_dllm_pretrained(
        fast_dllm_model_path=model_args.model_name_or_path,
        gla_config=gla_config,
        trust_remote_code=model_args.trust_remote_code,
    )
    student_model.config.use_cache = False

    if model_args.freeze_inherited_weights:
        _freeze_inherited_weights(student_model, model_args.model_name_or_path, model_args.trust_remote_code)

    _maybe_enable_gradient_checkpointing(student_model, distill_args.gradient_checkpointing)
    student_model.train()

    print("[Model] Loading frozen teacher model...")
    teacher_dtype = TORCH_DTYPE_MAP.get(distill_args.teacher_torch_dtype or "auto", None)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        distill_args.teacher_model_name_or_path,
        revision=distill_args.teacher_revision,
        trust_remote_code=distill_args.teacher_trust_remote_code,
        torch_dtype=teacher_dtype,
        low_cpu_mem_usage=True,
    )
    teacher_model.config.use_cache = False
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_devices = _string_to_devices(distill_args.teacher_visible_devices)
    student_devices = _string_to_devices(distill_args.student_visible_devices)

    if torch.cuda.is_available():
        print(f"[Model] Dispatching teacher to devices {teacher_devices or ['cpu']}...")
    teacher_model, teacher_device_map = _dispatch_parallel_model(
        teacher_model,
        teacher_devices,
        distill_args.teacher_max_memory_per_gpu,
        distill_args.teacher_offload_to_cpu,
    )

    if torch.cuda.is_available():
        print(f"[Model] Dispatching student to devices {student_devices or ['cpu']}...")
    student_model, student_device_map = _dispatch_parallel_model(
        student_model,
        student_devices,
        distill_args.student_max_memory_per_gpu,
        distill_args.student_offload_to_cpu,
    )

    teacher_primary_device = _primary_device_from_ids(teacher_devices)
    student_primary_device = _primary_device_from_ids(student_devices)

    pad_token_id = _resolve_pad_token_id(tokenizer, distill_args.pad_token_id_override)

    model_max_length = tokenizer.model_max_length if tokenizer.model_max_length and tokenizer.model_max_length > 0 else 4096
    train_dataset = prepare_dataset(
        tokenizer,
        data_args,
        distill_args.dataset_materialize_batch_size,
        model_max_length,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    persistent_workers = distill_args.dataloader_num_workers > 0
    loader_kwargs = {
        "batch_size": distill_args.per_device_train_batch_size,
        "shuffle": True,
        "drop_last": distill_args.dataloader_drop_last,
        "num_workers": distill_args.dataloader_num_workers,
        "pin_memory": distill_args.dataloader_pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": data_collator,
    }
    if persistent_workers and distill_args.dataloader_prefetch_factor > 0:
        loader_kwargs["prefetch_factor"] = distill_args.dataloader_prefetch_factor
    train_dataloader = DataLoader(train_dataset, **loader_kwargs)

    trainable_params = _gather_trainable_parameters(student_model)
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain in the student model.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=distill_args.learning_rate,
        betas=(distill_args.adam_beta1, distill_args.adam_beta2),
        eps=distill_args.adam_epsilon,
        weight_decay=distill_args.weight_decay,
    )

    updates_per_epoch = max(
        math.ceil(len(train_dataloader) / distill_args.gradient_accumulation_steps),
        1,
    )
    if distill_args.max_steps > 0:
        total_update_steps = distill_args.max_steps
        num_train_epochs = math.ceil(total_update_steps / updates_per_epoch)
    else:
        num_train_epochs = math.ceil(distill_args.num_train_epochs)
        total_update_steps = num_train_epochs * updates_per_epoch

    lr_scheduler = get_scheduler(
        distill_args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=distill_args.warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_fp16 = distill_args.fp16 and torch.cuda.is_available()
    use_bf16 = distill_args.bf16 and torch.cuda.is_available()

    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    layer_indices = None
    ce_weight = distill_args.ce_loss_weight
    hidden_weight = distill_args.hidden_loss_weight
    logits_weight = distill_args.logits_loss_weight

    global_step = 0
    completed_steps = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(num_train_epochs):
        if completed_steps >= total_update_steps:
            break
        for step, batch in enumerate(train_dataloader):
            if completed_steps >= total_update_steps:
                break

            input_ids = batch["input_ids"].to(student_primary_device, non_blocking=True)
            labels = batch.get("labels", input_ids).to(student_primary_device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is None:
                attention_mask = (batch["input_ids"] != pad_token_id).long()
            attention_mask_student = attention_mask.to(student_primary_device, non_blocking=True)
            attention_mask_teacher = attention_mask.to(teacher_primary_device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_fp16 or use_bf16, dtype=amp_dtype):
                student_outputs = student_model(
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                student_loss = student_outputs.loss if student_outputs.loss is not None else torch.zeros((), device=input_ids.device)

                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch["input_ids"].to(teacher_primary_device, non_blocking=True),
                        attention_mask=attention_mask_teacher,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )

                if layer_indices is None:
                    total_layers = min(len(student_outputs.hidden_states), len(teacher_outputs.hidden_states))
                    layer_indices = _select_layer_indices(
                        total_layers=total_layers,
                        max_layers=distill_args.max_distill_layers,
                        match_final_only=distill_args.match_final_only,
                    )

                hidden_loss = hidden_state_alignment_loss(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states,
                    attention_mask_student,
                    layer_indices,
                    match_embeddings=distill_args.match_embeddings,
                    normalize=distill_args.normalize_hidden_states,
                ) if hidden_weight > 0 else student_loss.new_zeros(())

                if logits_weight > 0:
                    teacher_logits = teacher_outputs.logits.to(student_outputs.logits.device)
                    logit_loss = logits_kl_loss(
                        student_outputs.logits,
                        teacher_logits,
                        temperature=distill_args.temperature,
                        attention_mask=attention_mask_student,
                    )
                else:
                    logit_loss = student_loss.new_zeros(())

                loss = (
                    ce_weight * student_loss
                    + hidden_weight * hidden_loss
                    + logits_weight * logit_loss
                )

            loss = loss / distill_args.gradient_accumulation_steps
            running_loss += loss.detach().item()

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % distill_args.gradient_accumulation_steps == 0:
                if distill_args.max_grad_norm is not None and distill_args.max_grad_norm > 0:
                    if use_fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, distill_args.max_grad_norm)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                completed_steps += 1
                global_step += 1

                if distill_args.logging_steps > 0 and global_step % distill_args.logging_steps == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / distill_args.logging_steps
                    print(
                        f"[Train] step={global_step} | epoch={epoch + 1} | "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e} | loss={avg_loss:.4f} | time={elapsed:.1f}s"
                    )
                    running_loss = 0.0
                    start_time = time.time()

                if distill_args.save_steps and distill_args.save_steps > 0:
                    if global_step % distill_args.save_steps == 0:
                        save_checkpoint(
                            student_model,
                            tokenizer,
                            distill_args.output_dir,
                            global_step,
                            distill_args.save_total_limit,
                            distill_args.save_safetensors,
                        )

    final_dir = distill_args.output_dir
    print(f"[Train] Training complete. Saving final student model to {final_dir}")
    student_model.save_pretrained(final_dir, safe_serialization=distill_args.save_safetensors)
    tokenizer.save_pretrained(final_dir)
    print("[Train] Done.")


if __name__ == "__main__":
    train()
