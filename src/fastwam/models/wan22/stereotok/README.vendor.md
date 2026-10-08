# Frozen StereoTok inference sources

- Wan VAE, default frozen CUDA inference runtime, Triton kernels: StereoTok main `4be840f97694e976714d549a4db2a53705384626`, files `wan/modules/vae2_2.py`, `stereotok/tokenizer_inference.py`, `stereotok/inference_kernels.py`.
- Standalone student/LAS2 trunk and strict student-v3 loader: Orion-0 MR70 `783b602d128a43814af32285e60a60f270ec2703`, `src/wam/models/modules/stereotok/`. Its LAS2 source is `8c97bd4c4da3712c2ac60003a23201dfdb5935f4`.
- No DA3, LPIPS, training teachers, iterative disparity decoder or training repository checkout is needed by this package. Matcher weights must be present in the deployment export. `timm` constructs its FasterNet backbone with `pretrained=False`.
- `LICENSE` and `LICENSE.Wan` preserve Apache-2.0 attribution. `las2/LICENSE` preserves LAS2's MIT attribution. Sources are included in Python distributions through `fastwam*`; notices are included as package data.

Local changes: relative package imports; camera-region argument forwarded through the newest eager/frozen inference routes; MR70's per-frame masks and camera-tile candidate restriction; strict Stage2/u8000 provenance admission. The newest runtime retains independent frozen module views and bounded matcher graph caches. This 384x320 integration has CPU contract tests; GPU compilation, full checkpoint strict-load and numerical comparisons remain server acceptance gates.
