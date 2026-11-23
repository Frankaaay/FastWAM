# Frozen StereoTok inference sources

- Wan VAE, frozen CUDA inference runtime and Triton kernels are vendored from StereoTok
  (`wan/modules/vae2_2.py`, `stereotok/tokenizer_inference.py`, `stereotok/inference_kernels.py`).
- The standalone student, LAS2-H matching trunk (LiteAnyStereo `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`)
  and the student-v3 loader are a teacher-free copy of the StereoTok student.
- No DA3, LPIPS, training teachers or StereoTok training checkout is needed by this package. Matcher
  weights are contained in the StereoTok export. `timm` constructs the FasterNet backbone with `pretrained=False`.
- `LICENSE` and `LICENSE.Wan` preserve Apache-2.0 attribution; `las2/LICENSE` preserves LAS2's MIT attribution.

Local changes: relative package imports; the camera-region argument is forwarded through the eager and frozen
inference routes; per-frame masks and camera-tile candidate restriction for multi-camera canvases; Stage-3
exports also load the shallow disparity readout (`metric_head`), which encoding does not call.
