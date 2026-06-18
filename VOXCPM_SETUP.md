# VoxCPM2 local provider

Shadow Reader now includes an optional `voxcpm` TTS provider backed by the
open-source VoxCPM2 model.

## Install

VoxCPM is a large local model stack. Install it only if you want to use the
local provider:

```bash
pip install -r requirements-voxcpm.txt
```

VoxCPM requires Python `>=3.10,<3.13`. The first request may download model
weights for `openbmb/VoxCPM2` and can take a while.

## Run

```bash
python app.py
```

Open the app, choose `VoxCPM2（本地开源）` in the provider selector, then generate
audio as usual. No API key is required.

## Configuration

The provider is lazy-loaded and configured through environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | Hugging Face model id or local model directory. |
| `VOXCPM_DEVICE` | `auto` | Runtime device: `auto`, `cpu`, `mps`, `cuda`, or `cuda:N`. |
| `VOXCPM_CACHE_DIR` | unset | Optional Hugging Face cache directory. |
| `VOXCPM_LOCAL_FILES_ONLY` | `false` | Use local files only; avoids network downloads. |
| `VOXCPM_LOAD_DENOISER` | `false` | Load the optional denoiser. |
| `VOXCPM_OPTIMIZE` | `false` | Enable VoxCPM/PyTorch optimization. Startup may be slower. |
| `VOXCPM_CFG_VALUE` | `2.0` | Guidance scale passed to `model.generate`. |
| `VOXCPM_INFERENCE_TIMESTEPS` | `10` | Diffusion inference steps. Higher may improve quality but is slower. |
| `VOXCPM_NORMALIZE` | `false` | Enable VoxCPM text normalization. |
| `VOXCPM_DENOISE` | `false` | Denoise prompt/reference audio when denoiser is loaded. |

Example:

```bash
VOXCPM_DEVICE=mps VOXCPM_OPTIMIZE=false python app.py
```
