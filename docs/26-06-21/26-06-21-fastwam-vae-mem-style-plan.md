# FastWAM VAE 的 MEM-style History Video 改造方案

## 0. 最终结论

我们采用的设计是：

> **VAE 输入端接收 `history_video + current_video`，在 VAE encoder 的中间 attention layers 中加入 MEM-style spatio-temporal attention；VAE 输出后立即切掉 history 对应的 latent，只把 current latent 传给原 FastWAM 的 Video DiT / Action DiT。**

因此，FastWAM 后端 DiT 的输入接口保持不变：

\[
z_{\mathrm{cur}} \rightarrow \mathrm{DiT}
\]

但这个 \(z_{\mathrm{cur}}\) 已经不是只由 current frame 独立编码得到，而是：

\[
z_{\mathrm{cur}}
=
\mathrm{CurrentSlice}
\left[
\mathrm{VAE}_{\mathrm{ST}}
\left(
[x_{\mathrm{hist}}, x_{\mathrm{cur}}]
\right)
\right]
\]

其中 `CurrentSlice` 表示：从 VAE 输出 latent 序列里只保留 current 对应的 latent timestep，history latent 不传给 DiT。

---

## 1. 参考 MEM 的核心思想

MEM 的 video encoder 做了三件事：

1. 输入是过去多帧 observation 和当前 observation；
2. 在视频 encoder 的多个中间层里交替执行 spatial attention 和 causal temporal attention；
3. 最后丢弃过去 timestep 的 observation tokens，只把 current timestep 的表示传给后续 VLA backbone。

迁移到 FastWAM 后，对应关系是：

| MEM | FastWAM-VAE 改造 |
|---|---|
| ViT image encoder | Wan/FastWAM convolutional Video VAE encoder |
| intermediate ViT layers | VAE encoder 中已有的 `AttentionBlock` 所在层 |
| spatial attention within each observation | 原 FastWAM VAE 的 spatial self-attention |
| causal temporal attention across observations | 新增 temporal attention branch |
| drop past observation tokens | VAE 输出后切掉 history latent |
| pass current tokens to VLA | 只把 current latent 传给 FastWAM DiT |

关键不是把 FastWAM VAE 改成 ViT，而是把 MEM 的 **space-time separable attention pattern** 移植到 FastWAM VAE 的 attention block 中。

---

## 2. FastWAM 原始 VAE 结构

FastWAM 使用的是 `WanVideoVAE38`，核心文件为：

```text
src/fastwam/models/wan22/wan_video_vae.py
```

初始化结构大致是：

```python
class WanVideoVAE38(WanVideoVAE):
    def __init__(self, z_dim=48, dim=160):
        ...
        self.model = VideoVAE38_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.temporal_downsample_factor = 4
        self.z_dim = z_dim
```

所以原始 VAE 的输入输出大致是：

\[
x \in \mathbb{R}^{B \times 3 \times T \times H \times W}
\]

\[
z \in \mathbb{R}^{B \times 48 \times T_{\mathrm{lat}} \times H/16 \times W/16}
\]

其中时间压缩倍率是 4。对于 Wan VAE 常见的时间组织：

\[
T = 4n + 1
\]

对应 latent 时间长度大致为：

\[
T_{\mathrm{lat}} = n + 1
\]

---

## 3. 原始 `AttentionBlock` 在做什么

FastWAM VAE 中的 `AttentionBlock` 代码结构是：

```python
class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)

        q, k, v = self.to_qkv(x).reshape(
            b * t, 1, c * 3, -1
        ).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        return x + identity
```

它把：

\[
[B,C,T,H,W]
\]

reshape 成：

\[
[BT,C,H,W]
\]

然后对每一帧内部的 \(H \times W\) 空间 token 做 self-attention。也就是说，原始 attention 是：

\[
\text{spatial attention within each timestep}
\]

不是 temporal attention。

---

## 4. 我们要改的核心：`AttentionBlock` → `SpatioTemporalAttentionBlock`

原来代码里如果存在：

```python
if scale in attn_scales:
    downsamples.append(AttentionBlock(out_dim))
```

我们改成：

```python
if scale in attn_scales:
    downsamples.append(SpatioTemporalAttentionBlock(out_dim))
```

对于 `Encoder3d_38` 里的：

```python
self.middle = nn.Sequential(
    ResidualBlock(out_dim, out_dim, dropout),
    AttentionBlock(out_dim),
    ResidualBlock(out_dim, out_dim, dropout),
)
```

改成：

```python
self.middle = nn.Sequential(
    ResidualBlock(out_dim, out_dim, dropout),
    SpatioTemporalAttentionBlock(out_dim),
    ResidualBlock(out_dim, out_dim, dropout),
)
```

原则是：

> **所有原来出现 spatial `AttentionBlock` 的地方，都替换成 `SpatioTemporalAttentionBlock`。**

如果当前使用的 VAE 配置只在 middle 有 attention，那么第一版只改 middle；如果启用了 `attn_scales` 并在 downsample/upsample stages 中插入多个 attention，则这些 intermediate attention layers 都要一起替换。这样才符合 MEM 的“temporal attention throughout intermediate layers”思想。

---

## 5. SpatioTemporalAttentionBlock 的内部结构

新的 block 保留原 spatial attention，并额外增加 temporal branch：

\[
x_s = x + W_o^s \mathrm{SpatialAttn}(W_qx, W_kx, W_vx)
\]

\[
x_{st}
=
x_s
+
\gamma
W_o^t
\mathrm{TemporalAttn}(W_qx_s, W_kx_s, W_vx_s)
\]

其中：

- \(W_q,W_k,W_v\)：来自原来的 `to_qkv`，复用 spatial attention 的 projection；
- \(W_o^s\)：原 spatial `proj`；
- \(W_o^t\)：新增 `temporal_proj`；
- \(\gamma\)：新增 `temporal_gate`，zero-init；
- temporal attention 使用 causal mask，让 current 只能看 past/current，不看未来。

推荐实现：

```python
class SpatioTemporalAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # FastWAM 原有参数：保持命名，方便加载原 checkpoint
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # 我们新增的 temporal output projection
        self.temporal_proj = nn.Conv2d(dim, dim, 1)

        # 我们新增的 gate：初始化为 0，初始不影响原模型
        self.temporal_gate = nn.Parameter(torch.zeros(()))
```

---

## 6. 为什么 QKV 可以共享，但 temporal_proj 要单独存在

`to_qkv` 的作用是把视觉 token 投影到 query/key/value 空间：

\[
Q = W_qx,\quad K = W_kx,\quad V = W_vx
\]

共享 `to_qkv` 的含义是：temporal attention 复用原 spatial attention 已经学好的视觉 token 比较空间。这样 current token 和 history token 可以在同一个表征坐标系里匹配。

但是 spatial attention 和 temporal attention 的目的不同：

\[
W_o^s:
\text{同一帧空间上下文}
\rightarrow
\text{当前 token residual}
\]

\[
W_o^t:
\text{历史时间上下文}
\rightarrow
\text{当前 token residual}
\]

因此我们新增 `temporal_proj`，而不是强行共享 spatial `proj`。

推荐初始化：

```python
def init_temporal_from_spatial(model):
    for m in model.modules():
        if isinstance(m, SpatioTemporalAttentionBlock):
            m.temporal_proj.weight.data.copy_(m.proj.weight.data)
            if m.proj.bias is not None:
                m.temporal_proj.bias.data.copy_(m.proj.bias.data)
```

调用顺序必须是：

```python
model.load_state_dict(state_dict, strict=False)
init_temporal_from_spatial(model)
```

原因是 `proj` 的预训练权重在 checkpoint 加载后才有效。`temporal_proj` 从 `proj` 拷贝初始化，之后独立训练。

---

## 7. Spatial attention 和 temporal attention 的 reshape 差异

### 7.1 Spatial attention

Spatial attention 的 token 组织是：

\[
[B,C,T,H,W]
\rightarrow
[BT,HW,C]
\]

含义是每一帧内部的空间 patch 互相 attend：

\[
x_{p,t}
\leftarrow
x_{p,t}
+
\sum_{p'}
\mathrm{softmax}
(q_{p,t}^{\top}k_{p',t})v_{p',t}
\]

### 7.2 Temporal attention

Temporal attention 的 token 组织是：

\[
[B,C,T,H,W]
\rightarrow
[BHW,T,C]
\]

含义是同一个空间 patch 沿时间维 attend：

\[
x_{p,t}
\leftarrow
x_{p,t}
+
\sum_{\tau \le t}
\mathrm{softmax}
(q_{p,t}^{\top}k_{p,\tau})v_{p,\tau}
\]

核心代码：

```python
def temporal_attn(self, x):
    b, c, t, h, w = x.shape

    x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
    x_2d = self.norm(x_2d)

    qkv = self.to_qkv(x_2d)
    qkv = qkv.reshape(b, t, 3 * c, h * w)
    qkv = qkv.permute(0, 1, 3, 2).contiguous()
    q, k, v = qkv.chunk(3, dim=-1)

    q = rearrange(q, 'b t n c -> (b n) 1 t c')
    k = rearrange(k, 'b t n c -> (b n) 1 t c')
    v = rearrange(v, 'b t n c -> (b n) 1 t c')

    causal_mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
    causal_mask = causal_mask[None, None, :, :]

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=causal_mask,
    )

    out = out.squeeze(1)
    out = rearrange(out, '(b n) t c -> b c t h w',
                    b=b, n=h * w, h=h, w=w)

    out_2d = rearrange(out, 'b c t h w -> (b t) c h w')
    out_2d = self.temporal_proj(out_2d)
    out = rearrange(out_2d, '(b t) c h w -> b c t h w', b=b, t=t)

    return out
```

最终 forward：

```python
def forward(self, x):
    x = self.spatial_attn(x)

    if x.shape[2] > 1:
        temp = self.temporal_attn(x)
        x = x + torch.tanh(self.temporal_gate) * temp

    return x
```

---

## 8. VAE 输入输出数据流

原始 FastWAM：

```text
current_video
    ↓
Wan/FastWAM VAE
    ↓
z_cur
    ↓
Video DiT / Action DiT
```

我们的版本：

```text
history_video + current_video
    ↓ concat along time
memory_video
    ↓
Spatio-Temporal VAE Encoder
    ↓
z_mem = [z_hist_like, z_cur]
    ↓ slice current latent only
z_cur
    ↓
原 FastWAM Video DiT / Action DiT
```

也就是说，DiT 不接收 history latent。history 只在 VAE encoder 内部通过 temporal attention 被压进 current latent。

---

## 9. current latent 的切片方式

假设：

```python
history_video: [B, 3, T_hist, H, W]
current_video: [B, 3, T_cur,  H, W]
```

拼接：

```python
video_mem = torch.cat([history_video, current_video], dim=2)
```

编码：

```python
z_mem = vae.encode(video_mem)
```

如果 \(T_{\mathrm{hist}}\) 是 4 的倍数：

```python
def keep_current_latent(z_mem, T_hist, temporal_downsample_factor=4):
    assert T_hist % temporal_downsample_factor == 0
    hist_latents = T_hist // temporal_downsample_factor
    return z_mem[:, :, hist_latents:]
```

最干净的设置是：

```text
history_video: 4K frames
current_video: 1 frame
total_video:   4K + 1 frames
```

例如：

```text
history_video: [B, 3, 16, H, W]
current_video: [B, 3, 1,  H, W]
video_mem:     [B, 3, 17, H, W]
z_mem:         [B, 48, 5, H/16, W/16]
z_cur:         [B, 48, 1, H/16, W/16]
```

切片：

```python
z_cur = z_mem[:, :, -1:]
```

这个 \(z_{\mathrm{cur}}\) 的 latent slot 是 current 的，但其信息来源已经包含 history。

---

## 10. 训练方案

### 10.1 参数初始化

加载原始 VAE checkpoint：

```python
model.load_state_dict(state_dict, strict=False)
```

新增参数不会在 checkpoint 中出现，因此用 `strict=False`。

然后拷贝 temporal projection：

```python
init_temporal_from_spatial(model)
```

打开新增参数训练：

```python
model.requires_grad_(False)

for name, p in model.named_parameters():
    if (
        "temporal_proj" in name
        or "temporal_gate" in name
        or "temporal_pos" in name
        or "temporal_lora" in name
    ):
        p.requires_grad_(True)
```

### 10.2 第一阶段：稳定适配

训练：

```text
temporal_proj
temporal_gate
别忘了temporal position embedding
FastWAM 下游 action/video 模块
```

冻结：

```text
原 VAE conv / resblock / to_qkv / spatial proj
```

这样初始时 gate 为 0，模型行为等价于原 FastWAM。训练过程中 gate 逐渐打开，history 信息进入 current latent。

### 10.3 第二阶段：增强表达力

如果第一阶段 history 信息不足，可以解冻所有层

推荐形式：

\[
W_q^t = W_q^s + \Delta W_q
\]

\[
W_k^t = W_k^s + \Delta W_k
\]

\[
W_v^t = W_v^s + \Delta W_v
\]

其中 \(\Delta W\) 用 LoRA 或 zero-init adapter 表示。

### 10.4 训练 loss

FastWAM 原有 loss 保持不变。因为下游 DiT 的输入仍然是 `z_cur`，训练标签仍然对应 current 片段。

可选辅助 loss：

\[
\mathcal{L}_{\mathrm{latent}}
=
\|z_{\mathrm{cur}}^{\mathrm{mem}} - z_{\mathrm{cur}}^{\mathrm{orig}}\|_2
\]

早期可以用小权重约束 latent distribution，避免 temporal branch 打开过快导致 DiT 输入分布偏移。


## 11. 最终方案一句话

最终方案是：

> **将 history video 与 current video 在时间维拼接后输入 FastWAM 的 Wan VAE；把 VAE encoder 中所有已有 spatial `AttentionBlock` 替换为 MEM-style `SpatioTemporalAttentionBlock`，其中 spatial branch 保持原权重，temporal branch 复用 `to_qkv`，新增 `temporal_proj` 和 `temporal_gate`，`temporal_proj` 从原 spatial `proj` 拷贝初始化，gate zero-init；VAE 输出后切掉 history 对应的 latent，只保留 current latent 送入原 FastWAM DiT。**

这样做实现了三点：

1. history 信息在 VAE intermediate layers 中逐层注入 current representation；
2. DiT 输入接口保持不变，计算量和 token 数量基本不增加；
3. 通过 copy init 和 zero gate 保留原 FastWAM/Wan VAE 的预训练 latent space，训练风险可控。
