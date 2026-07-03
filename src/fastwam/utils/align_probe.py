from collections import Counter
from typing import Any, Optional

import torch
from torch import nn

from .logging_config import get_logger

logger = get_logger(__name__)


def log_parameter_alignment(
    model: nn.Module,
    *,
    is_main_process: bool,
    sample_limit: int = 10,
) -> None:
    if not is_main_process:
        return

    total_count = 0
    zero_numel_count = 0
    checked_count = 0
    unaligned_count = 0
    remainder_counts: Counter[int] = Counter()
    unaligned_samples = []

    for name, param in model.named_parameters():
        total_count += 1
        numel = param.numel()
        if numel == 0:
            zero_numel_count += 1
            continue

        checked_count += 1
        remainder = int(param.data_ptr() % 16)
        remainder_counts[remainder] += 1

        if remainder != 0:
            unaligned_count += 1
            if len(unaligned_samples) < sample_limit:
                unaligned_samples.append((name, remainder, tuple(param.shape), numel, param.dtype))

    remainder_desc = _format_remainder_counts(remainder_counts)
    logger.info(
        "[align-probe] param_alignment total_params=%d checked_params=%d zero_numel_params=%d "
        "unaligned_params=%d ptr_mod16_distribution={%s}",
        total_count,
        checked_count,
        zero_numel_count,
        unaligned_count,
        remainder_desc,
    )

    for index, (name, remainder, shape, numel, dtype) in enumerate(unaligned_samples, start=1):
        logger.info(
            "[align-probe] param_alignment_sample index=%d name=%s ptr_mod16=%d shape=%s numel=%d dtype=%s",
            index,
            name,
            remainder,
            shape,
            numel,
            dtype,
        )


def register_linear_activation_alignment_hooks(
    model: nn.Module,
    *,
    is_main_process: bool,
    max_layers: int = 8,
) -> None:
    if not is_main_process:
        return

    registered_count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        qualified_name = module_name or "<root>"
        _register_one_linear_hook(qualified_name, module)
        registered_count += 1
        if registered_count >= max_layers:
            break

    logger.info(
        "[align-probe] act_alignment_hooks registered=%d max_layers=%d",
        registered_count,
        max_layers,
    )


def _register_one_linear_hook(module_name: str, module: nn.Linear) -> None:
    handle_ref = {}

    def hook(layer: nn.Linear, inputs: Any) -> None:
        # 每个 hook 只记录第一次触发，避免训练日志被每步刷屏。
        handle = handle_ref.pop("handle", None)
        if handle is not None:
            handle.remove()

        input_tensor = _first_tensor(inputs)
        input_ptr_mod16 = _tensor_ptr_mod16(input_tensor)
        input_shape = tuple(input_tensor.shape) if input_tensor is not None else "none"
        input_contiguous = input_tensor.is_contiguous() if input_tensor is not None else "none"
        weight_ptr_mod16 = _tensor_ptr_mod16(layer.weight)

        logger.info(
            "[align-probe] act_alignment layer=%s input_ptr_mod16=%s input_shape=%s "
            "input_contiguous=%s weight_ptr_mod16=%s",
            module_name,
            input_ptr_mod16,
            input_shape,
            input_contiguous,
            weight_ptr_mod16,
        )

    handle_ref["handle"] = module.register_forward_pre_hook(hook)


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_ptr_mod16(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "none"
    if tensor.numel() == 0:
        return "zero_numel"
    return str(int(tensor.data_ptr() % 16))


def _format_remainder_counts(remainder_counts: Counter[int]) -> str:
    if not remainder_counts:
        return "none"
    return ", ".join(f"{remainder}:{count}" for remainder, count in sorted(remainder_counts.items()))
