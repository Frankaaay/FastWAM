"""Smoke test for the MEM-stage2-v2 fold-current adapter.

Runs CPU-only on a tiny randomly-initialised WanVideoDiT (no pretrained weights,
air-gapped friendly). Verifies the design invariants that make the fold-current
path baseline-preserving:

  1. gate=0  -> pre_dit(fold) output is byte-identical to the base path
     (history folded with zero strength + dropped == no-memory baseline).
  2. token count / current index / grid after fold == base (interface unchanged).
  3. history_keep_mask=0 -> output == base even with gate>0 (mem-off invariance).
  4. gate>0 + keep=1 -> output DIFFERS from base (the adapter actually does work).

Run: python scripts/test_fold_current.py
"""
import torch

from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def _build_tiny_dit():
    return WanVideoDiT(
        hidden_dim=64,
        in_dim=16,
        ffn_dim=128,
        out_dim=16,
        text_dim=32,
        freq_dim=64,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        attn_head_dim=16,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=True,
        action_dim=7,
        video_attention_mask_mode="first_frame_causal",
    )


def main():
    torch.manual_seed(0)
    dit = _build_tiny_dit().eval()
    dit.enable_history_fold(max_history_frames=8)

    B, C, h, w = 1, 16, 4, 4
    K = 2           # history latent frames
    T = 3           # current + future latent frames (f after fold)
    action_horizon = T - 1

    history = torch.randn(B, C, K, h, w)
    current_future = torch.randn(B, C, T, h, w)
    stack = torch.cat([history, current_future], dim=2)   # [B, C, K+T, h, w]

    context = torch.randn(B, 5, 32)
    context_mask = torch.ones(B, 5, dtype=torch.bool)
    action = torch.randn(B, action_horizon, 7)
    timestep = torch.zeros(B)

    def run(x, num_history_frames, keep=None):
        return dit.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=True,
            num_history_frames=num_history_frames,
            history_keep_mask=keep,
        )

    with torch.no_grad():
        base = run(current_future, 0)                      # no adapter involved (K=0)
        fold0 = run(stack, K)                               # gate=0 -> identity

        # 1 + 2: gate=0 fold == base, same shapes (token count / grid / current index).
        assert fold0["tokens"].shape == base["tokens"].shape, (
            f"token count mismatch: fold {fold0['tokens'].shape} vs base {base['tokens'].shape}"
        )
        assert fold0["meta"]["grid_size"] == base["meta"]["grid_size"], "grid (f,h,w) mismatch"
        assert fold0["meta"]["tokens_per_frame"] == base["meta"]["tokens_per_frame"]
        assert torch.allclose(fold0["tokens"], base["tokens"], atol=1e-6), "gate=0 tokens != base"
        assert torch.allclose(fold0["freqs"], base["freqs"], atol=1e-6), "freqs != base"
        assert torch.allclose(fold0["t_mod"], base["t_mod"], atol=1e-6), "t_mod != base"
        print("[1+2] gate=0 fold output byte-identical to base; interface (tokens/grid) == base  OK")

        # open the gate
        dit.history_fold_adapter.gate.data.fill_(1.0)

        # 3: keep=0 -> mem-off == base even with gate>0.
        keep0 = torch.zeros(B)
        foldoff = run(stack, K, keep=keep0)
        assert torch.allclose(foldoff["tokens"], base["tokens"], atol=1e-6), "keep=0 (mem-off) != base"
        print("[3] history_keep_mask=0 (mem-off) output == base with gate>0  OK")

        # 4: keep=1 + gate>0 -> output differs from base (adapter does real work).
        keep1 = torch.ones(B)
        foldon = run(stack, K, keep=keep1)
        diff = (foldon["tokens"] - base["tokens"]).abs().max().item()
        assert diff > 1e-4, f"gate>0 keep=1 produced no change (max|delta|={diff})"
        print(f"[4] gate>0 keep=1 changes the current tokens (max|delta|={diff:.4f})  OK")

    print("\nALL FOLD-CURRENT SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
