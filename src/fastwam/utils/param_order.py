"""Utilities for optimizer parameter ordering."""


def reorder_params_for_alignment(params: list, *, alignment_bytes: int = 16) -> tuple[list, list]:
    """Move parameters with non-aligned lengths to the tail, preserving order."""
    if alignment_bytes <= 0:
        raise ValueError(f"alignment_bytes must be positive, got {alignment_bytes}")

    aligned = []
    tail = []
    for param in params:
        element_size = param.element_size()
        if element_size <= 0:
            raise ValueError(f"Parameter element_size must be positive, got {element_size}")
        align_elements = alignment_bytes // element_size
        if align_elements <= 0:
            raise ValueError(
                "alignment_bytes must be at least the parameter element size: "
                f"alignment_bytes={alignment_bytes} element_size={element_size}"
            )

        if param.numel() % align_elements == 0:
            aligned.append(param)
        else:
            tail.append(param)

    return aligned + tail, tail


def log_param_order_tail(
    tail,
    named_lookup,
    *,
    logger,
    is_main_process: bool,
    alignment_bytes: int = 16,
) -> None:
    """Log parameters moved to the tail by alignment-aware ordering."""
    if not is_main_process:
        return

    logger.info("param-order tail_count=%d alignment_bytes=%d", len(tail), alignment_bytes)
    for index, param in enumerate(tail, start=1):
        name = named_lookup.get(id(param))
        if name is None:
            try:
                name = named_lookup.get(param)
            except TypeError:
                name = None
        if name is None:
            name = "<unnamed>"
        logger.info(
            "param-order tail index=%d name=%s numel=%d shape=%s",
            index,
            name,
            param.numel(),
            tuple(param.shape),
        )
