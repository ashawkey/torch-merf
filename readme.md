# torch-MeRF

An unofficial pytorch implementation of [MeRF: Memory-Efficient Radiance Fields for Real-time View Synthesis in Unbounded Scenes](https://merf42.github.io/).

We support exporting almost lossless baked assets for the real-time web renderer.

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
# train (and test after finishing)
python main.py data/bonsai/ --workspace trial_bonsai --enable_cam_center --downscale 4

# test (export video and baked assets)
python main.py data/bonsai/ --workspace trial_bonsai --enable_cam_center --downscale 4 --test

# web renderer
# use VS Code to host the folder and open ./renderer/renderer.html
# follow the instructions and add the baked assets path as URL parameters to start rendering.
# for example:
http://localhost:5500/renderer/renderer.html?dir=trial_bonsai/assets
http://localhost:5500/renderer/renderer.html?dir=trial_bonsai/assets&quality=low # phone, low, medium, high
```

Please check the `scripts` directory for more examples on common datasets, and check `main.py` for all options.

### Implementation Notes

#### Modification of web renderer
The web renderer is slightly modified from the official version:

* Frequency encoding convention (viewdependency.glsl):
    ```bash
    # original
    x, sin(x), sin(2x), sin(4x), ..., cos(x), cos(2x), cos(4x), ...
    # current
    x, sin(x), cos(x), sin(2x), cos(2x), sin(4x), cos(4x), ...
    ```

* Interpolation alignment of sparse grid (fragment.glsl):
    ```cpp
    // original
    vec3 posSparseGrid = (z - minPosition) / voxelSize - 0.5;
    // current
    vec3 posSparseGrid = (z - minPosition) / voxelSize;
    ```


#### Lossless baking
The baking can be lossless since the baked assets' resolution is the same as the network's resolution, 
but **interpolation must happen after all non-linear functions (i.e., MLP)**. 

This makes the usual hashgrid + MLP combination invalid as

$$
\text{MLP}(\sum_i(w_i * x_i)) \ne \sum_i(w_i * \text{MLP}(x_i))
$$

(using a single linear layer should be able to work though? but the paper uses 2 layers with 64 hidden dims...)

In this implementation we have to manually perform bilinear/trilinear interpolation in torch, and query the 4/8 corners of the grid for each sampling point, which is quite inefficient...

#### Interpolation alignment

OpenGL's `texture()` behaves like the `F.interpolate(..., align_corners=False)`.
It seems the sparse grid uses `align_corners=True` convention, while the triplane uses `align_corners=False` convention... but maybe I'm wrong somewhere, since I have to modify the web renderer to make it work.

#### gridencoder
The API is slightly abused for convenience, we need to pass in values in the range of [0, 1] (the `bound` parameter is removed).

### Performance reference 

|        | Bonsai | Counter | Kitchen | Room | Bicycle | Garden | Stump |
| ---    | --- | --- | --- | --- | --- | --- | --- |
| MipNeRF 360 (~days)              | 33.46 | 29.55 | 32.23 | 31.63 | 24.57 | 26.98 | 26.40 | 
| baseline-nerfacto (~12 minutes)  | 31.10 | 26.65 | 30.61 | 31.44 | 23.74 | 25.31 | 25.48 |

Ours are tested on a V100. 
Please check the commands under `scripts/` to reproduce.

### Acknowledgement
This repository is based on:
* [torch-ngp](https://github.com/ashawkey/torch-ngp)
* [DearPyGui](https://github.com/hoffstadt/DearPyGui)
* [nerfstudio](https://github.com/nerfstudio-project/nerfstudio)
* [nerfacc](https://github.com/KAIR-BAIR/nerfacc)