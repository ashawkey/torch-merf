# torch-MeRF

An unofficial pytorch implementation of [MeRF: Memory-Efficient Radiance Fields for Real-time View Synthesis in Unbounded Scenes](https://merf42.github.io/).

### Features
* Piecewise linear contraction.
* Exporting baked assets fully compatible to the official web renderer.

# Install

```bash
git clone https://github.com/ashawkey/torch-merf.git
cd torch-merf
```

### Install with pip
```bash
pip install -r requirements.txt
```

### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
However, this may be inconvenient sometimes.
Therefore, we also provide the `setup.py` to build each extension:
```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

### Tested environments
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.

# Usage

We majorly support COLMAP dataset like [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).
Please download and put them under `./data`.

For custom datasets:
```bash
# prepare your video or images under /data/custom, and run colmap (assumed installed):
python scripts/colmap2nerf.py --video ./data/custom/video.mp4 --run_colmap # if use video
python scripts/colmap2nerf.py --images ./data/custom/images/ --run_colmap # if use images
```

### Basics
First time running will take some time to compile the CUDA extensions.
```bash
# train
python main.py data/bonsai/ --workspace trial_bonsai --enable_cam_center --downscale 4 -O2

# export baked assets

# web renderer

```

Please check the `scripts` directory for more examples on common datasets, and check `main.py` for all options.

### Performance reference 

|        | Bonsai | Counter | Kitchen | Room | Bicycle | Garden | Stump |
| ---    | --- | --- | --- | --- | --- | --- | --- |
| MipNeRF 360 (~days)          | 33.46 | 29.55 | 32.23 | 31.63 | 24.57 | 26.98 | 26.40 | 
| ours-ngp (~8 minutes)        | 28.99 | 25.18 | 26.42 | 28.58 | 21.31 | 23.70 | 22.73 |
| ours-nerfacto (~12 minutes)  | 31.10 | 26.65 | 30.61 | 31.44 | 23.74 | 25.31 | 25.48 |

Ours are tested on a V100. 
Please check the commands under `scripts/` to reproduce.

### Acknowledgement
This repository is based on:
* [torch-ngp](https://github.com/ashawkey/torch-ngp)
* [DearPyGui](https://github.com/hoffstadt/DearPyGui)
* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
* [nerfacc](https://github.com/KAIR-BAIR/nerfacc)