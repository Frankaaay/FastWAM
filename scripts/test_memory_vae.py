"""Standalone sanity checks for the MEM-style memory VAE additions.

Run on the server (needs only torch + einops, no checkpoint / dataset):

    python scripts/test_memory_vae.py

It verifies:
  1. SpatioTemporalAttentionBlock preserves shape.
  2. With temporal_gate == 0 it is byte-equivalent to the original AttentionBlock
     (so loading a pretrained ckpt is a no-op at step 0).
  3. encode_memory returns the correct current-latent temporal length.
  4. init_temporal_from_spatial copies proj -> temporal_proj.
  5. set_vae_memory_trainable unfreezes only the temporal params.
  6. gradients flow into temporal params through encode_memory.
"""

import torch

from fastwam.models.wan22.wan_video_vae import (
    AttentionBlock,
    SpatioTemporalAttentionBlock,
    VideoVAE38_,
    WanVideoVAE38,
    init_temporal_from_spatial,
    set_vae_memory_trainable,
    MEMORY_PARAM_KEYS,
)


def test_shape_preserved():
    blk = SpatioTemporalAttentionBlock(dim=16).eval()
    x = torch.randn(2, 16, 5, 8, 8)
    y = blk(x)
    assert y.shape == x.shape, y.shape
    print("[ok] shape preserved:", tuple(y.shape))


def test_equivalent_to_spatial_at_init():
    dim = 16
    spatial = AttentionBlock(dim).eval()
    st = SpatioTemporalAttentionBlock(dim).eval()
    # copy the shared spatial params so the spatial branch is identical
    st.norm.load_state_dict(spatial.norm.state_dict())
    st.to_qkv.load_state_dict(spatial.to_qkv.state_dict())
    st.proj.load_state_dict(spatial.proj.state_dict())
    # temporal_gate is 0 -> temporal branch contributes nothing
    x = torch.randn(2, dim, 5, 8, 8)
    with torch.no_grad():
        y_spatial = spatial(x)
        y_st = st(x)
    max_diff = (y_spatial - y_st).abs().max().item()
    assert max_diff < 1e-5, max_diff
    print("[ok] ST == spatial at init, max|diff| =", max_diff)


def test_encode_memory_shape():
    # small model so it runs on CPU quickly
    model = VideoVAE38_(dim=16, z_dim=4, dec_dim=16, use_temporal_attention=True).eval()
    scale = [torch.zeros(4), torch.ones(4)]
    T_hist, T_cur = 16, 9          # total 25, %4==1
    x = torch.randn(1, 3, T_hist + T_cur, 32, 32)
    z = model.encode_memory(x, scale, num_current_frames=T_cur)
    exp_t = 1 + (T_cur - 1) // 4   # = 3
    assert z.shape[0] == 1 and z.shape[1] == 4 and z.shape[2] == exp_t, z.shape
    print("[ok] encode_memory current-latent shape:", tuple(z.shape))


def test_init_and_trainable():
    model = VideoVAE38_(dim=16, z_dim=4, dec_dim=16, use_temporal_attention=True)
    # set a recognizable proj weight, then copy
    for m in model.modules():
        if isinstance(m, SpatioTemporalAttentionBlock):
            m.proj.weight.data.normal_()
    init_temporal_from_spatial(model)
    for m in model.modules():
        if isinstance(m, SpatioTemporalAttentionBlock):
            assert torch.equal(m.temporal_proj.weight.data, m.proj.weight.data)
    print("[ok] init_temporal_from_spatial copied proj -> temporal_proj")

    vae = WanVideoVAE38(z_dim=4, dim=16, use_temporal_attention=True)
    trainable = set_vae_memory_trainable(vae)
    n_train = sum(p.requires_grad for p in vae.parameters())
    assert n_train == len(trainable) and n_train > 0
    for name, p in vae.named_parameters():
        if p.requires_grad:
            assert any(k in name for k in MEMORY_PARAM_KEYS), name
    print(f"[ok] only {n_train} temporal params unfrozen")


def test_grad_flows():
    model = VideoVAE38_(dim=16, z_dim=4, dec_dim=16, use_temporal_attention=True)
    init_temporal_from_spatial(model)
    # open the gate a little so there is a temporal signal to backprop
    for m in model.modules():
        if isinstance(m, SpatioTemporalAttentionBlock):
            m.temporal_gate.data.fill_(0.5)
    scale = [torch.zeros(4), torch.ones(4)]
    x = torch.randn(1, 3, 25, 32, 32, requires_grad=False)
    z = model.encode_memory(x, scale, num_current_frames=9)
    z.sum().backward()
    grads = {
        name: (p.grad is not None and p.grad.abs().sum().item() > 0)
        for name, p in model.named_parameters()
        if "temporal_proj" in name or "temporal_gate" in name
    }
    assert all(grads.values()), grads
    print("[ok] gradients reach temporal params:", list(grads))


if __name__ == "__main__":
    torch.manual_seed(0)
    test_shape_preserved()
    test_equivalent_to_spatial_at_init()
    test_encode_memory_shape()
    test_init_and_trainable()
    test_grad_flows()
    print("\nAll memory-VAE sanity checks passed.")
