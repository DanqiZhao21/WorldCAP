# WoTE controller module

This package contains the controller trajectory embedding used by the WoTE
latent world model and the minimal scripts required to regenerate controller
style bundles.

Runtime data is kept outside Git under the repository's `CtrlNew` symlink:

```text
CtrlNew/controller/ref_trajs/
CtrlNew/controller/bundles/{64,128,256,1024}/
```

The style-observation bank and planner-candidate bank have separate roles:

- `ref_trajs/Anchors_Original_1024_centered.npy` plus the 1024 bundle condition
  the world model through `ControllerEmbedding`.
- The 256 bundle supplies style-matched executed rollouts aligned with the 256
  planner candidates.

`generate_controller_bundle_clean.py` is the preferred generator. `genTest.py`
is retained because it owns the simulator and style definitions used by the
clean generator. `downsample_controller_bundle_refs.py` derives smaller bundles.
