#!/usr/bin/env python
# ----------------------------------------------------------------------------
# precompute_libero_plus_embeds.py — 为 LIBERO-plus 的全部任务指令预计算 T5 文本
# embedding,落到 eval 期望的缓存目录(离线、单卡、几分钟)。
#
# 为什么需要它:FastWAM 用 load_text_encoder=false(气隙无 T5),eval 时按
# sha256(prompt) 直接取预计算缓存。标准 LIBERO 只有 40 条指令已缓存;LIBERO-plus
# 把扰动参数编码进了 task.language(如 "...place it on the plate view 323 0 100 0
# 0 initstate 0"、"...light 48"、Language factor 则是 LLM 改写句)。按 LIBERO-plus
# 的 drop-in 协议(与 robustness paper 对齐),原样喂 task.language,因此这里为
# 所有唯一 task.language 预计算 embedding。
#
# 复用 scripts/precompute_text_embeds.py 的 encoder/tokenizer/保存约定,保证 hash
# key 与训练/eval 完全一致(同 DEFAULT_PROMPT 模板、同 t5_len128.wan22ti2v5b.pt
# 文件名、同 {context, mask} payload)。已存在的条目自动跳过。
#
# 用法:
#   source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastwam
#   export DIFFSYNTH_MODEL_BASE_PATH=$(pwd)/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true
#   export MUJOCO_GL=egl
#   CUDA_VISIBLE_DEVICES=0 python experiments/libero/precompute_libero_plus_embeds.py
# ----------------------------------------------------------------------------
import hashlib
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs  # noqa: E402
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer  # noqa: E402
from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from scripts.precompute_text_embeds import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LEN,
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_MODEL_ID,
    _atomic_torch_save,
    _model_id_to_enc_id,
)

register_default_resolvers()

from libero.libero import benchmark  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    ctx_len = DEFAULT_CONTEXT_LEN
    enc_id = _model_id_to_enc_id(DEFAULT_MODEL_ID)
    cache_dir = Path(os.path.expanduser("~/projects/FastWAM/data/text_embeds_cache/libero"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) 枚举全部唯一 task.language(原样,含扰动后缀)
    bdict = benchmark.get_benchmark_dict()
    seen, langs = set(), []
    for s in SUITES:
        ts = bdict[s]()
        for i in range(ts.n_tasks):
            lang = ts.get_task(i).language
            if lang not in seen:
                seen.add(lang)
                langs.append(lang)
    print(f"[enum] unique task.language across {SUITES}: {len(langs)}", flush=True)

    # 2) 过滤已缓存
    todo = []
    for lang in langs:
        prompt = DEFAULT_PROMPT.format(task=lang)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        fp = cache_dir / f"{hashed}.t5_len{ctx_len}.{enc_id}.pt"
        if not fp.exists():
            todo.append((prompt, fp))
    print(f"[plan] to-encode={len(todo)} already-cached={len(langs) - len(todo)}", flush=True)
    if not todo:
        print("[done] nothing to encode.")
        return

    # 3) 构建 T5 encoder + tokenizer(与 precompute_text_embeds 完全一致)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    _, text_config, _, tok_config = _resolve_configs(
        model_id=DEFAULT_MODEL_ID,
        tokenizer_model_id=DEFAULT_TOKENIZER_MODEL_ID,
        redirect_common_files=True,
    )
    text_config.download_if_necessary()
    tok_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=dtype, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tok_config.path, seq_len=ctx_len, clean="whitespace")

    # 4) 批量编码 + 落盘
    over_len = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(todo), DEFAULT_BATCH_SIZE), desc="encoding", unit="batch"):
            batch = todo[start : start + DEFAULT_BATCH_SIZE]
            prompts = [b[0] for b in batch]
            ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            over_len += int(mask.all(dim=1).sum().item())
            context = text_encoder(ids, mask)
            for i, (_prompt, fp) in enumerate(batch):
                payload = {
                    "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                }
                _atomic_torch_save(payload, fp)

    n_cached = len(list(cache_dir.glob(f"*.t5_len{ctx_len}.{enc_id}.pt")))
    print(f"[done] encoded={len(todo)} over_length(no_pad)={over_len} total_in_cache={n_cached}", flush=True)


if __name__ == "__main__":
    main()
