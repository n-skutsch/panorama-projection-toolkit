# Panorama Projection Toolkit

GPU-accelerated Python toolkit for projecting and reprojecting perspective views, cubemaps, and equirectangular panoramas. The toolkit provides fast, fully vectorized implementations of the following transformations:
- equirectangular panorama → perspective view
- perspective view → equirectangular panorama
- perspective view → perspective view
- cubemaps → equirectangular panorama

It supports both CPU execution through NumPy/SciPy and GPU acceleration through CuPy/CUDA with the same API.


## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- OpenCV
- Optional: CuPy + CUDA

### Installation from pip

```bash
pip install panorama_projection_toolkit
```

### Manual installation

Clone the repository:

```bash
git clone <repository-url>
cd panorama_projection_toolkit
```

Build the package:

```bash
python -m build
```

Install the wheel:

```bash
pip install dist/panorama_projection_toolkit-VERSION-py3-none-any.whl
```

### Optional: GPU Acceleration

The toolkit automatically uses CuPy if it is installed and CUDA is available. Otherwise, it falls back to NumPy/SciPy automatically. The mode can be set manually be setting the environment variable `PPT_GPU`.

1. Verify CUDA compatibility: https://developer.nvidia.com/cuda/gpus
2. Download and install CUDA Toolkit: https://developer.nvidia.com/cuda/toolkit
3. Install the matching CuPy build for your CUDA version: https://docs.cupy.dev/en/stable/install.html
4. Set the environment variable `PPT_GPU` to `0` for CPU mode (NumPy/SciPy) and `1` for GPU mode (CuPy).


## Supported Formats

| Format     | Read | Write | Typical data type |
| ---------- | ---- | ----- | ----------------- |
| JPG / JPEG | ✔   | ✔   | uint8             |
| PNG        | ✔   | ✔   | uint8             |
| EXR        | ✔   | ✔   | float32           |

Internally, most computations use `float32`. The output image is restored to the original input data type whenever possible. OpenCV is used internally, so color images use BGR channel ordering.

OpenEXR support requires the following environment variable to be set before importing OpenCV:

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
```

This is already handled automatically by the package.


## Coordinate System and Panorama Convention

Equirectangular panoramas are interpreted as:
- horizontal axis → longitude θ ∈ [-π, π]
- vertical axis   → latitude  φ ∈ [-π/2, π/2]

The panorama origin corresponds to:
- center → north-facing direction
- top    → zenith
- bottom → nadir

The toolkit assumes a right-handed ENU coordinate system:
- X → East
- Y → North
- Z → Up

Euler angles are interpreted as pitch, roll, and yaw with rotations around:
- pitch → x-axis
- roll  → y-axis
- yaw   → z-axis

A yaw angle of 0° is equivalent to facing north. All angles are specified in degrees.


## Sampling Methods

| Sampling Method        | Application                     |
| ---------------------- | ------------------------------- |
| Nearest-neighbor       | 6-view cubemap stitching        |
| Bilinear interpolation | All other projection operations |


## Public API

```python
from panorama_projection_toolkit import (
    load_image,
    save_image,
    pano_to_view,
    view_to_pano,
    view_to_view,
    cubemaps_to_pano
)
```

### load_image

```python
load_image(file_path)
```

Loads an image from the specified file path. EXR files are typically loaded as floating-point arrays (e.g., `float32`), whereas all other file types are typically loaded as uint8 arrays. The image is returned as a NumPy array of shape `(height, width)` for single-channel images and `(height, width, channels)` for multi-channel images. For color images, the color channels are in BGR order. Non-EXR grayscale images are also loaded as 3-channel BGR images. In order to load images from EXR files, the environment variable `OPENCV_IO_ENABLE_OPENEXR` must be set to `1` before importing `cv2`.

Parameters:

| Parameter | Description                                  |
| --------- | -------------------------------------------- |
| file_path | The file path at which the image is located. |

Example:

```python
image = load_image('image.png')
```

### save_image

```python
save_image(file_path, image)
```

Saves an image to the specified file path. The image is expected to be a NumPy array of shape `(height, width)` for single-channel images and `(height, width, channels)` with color channels in BGR order for multi-channel images. If the file extension is either JPG/JPEG or PNG and the image is not an `uint8` array, it is converted to an `uint8` array before saving it. If the file extension is EXR and the image is not a `float32` array, it is converted to a `float32` array before saving it. These conversions may lead to unexpected results. The pixel values of `float` images are expected to fall within the range of `[0, 1]`. The pixel values of `uint8` images are expected to fall within the range of `[0, 255]`. In order to save images to EXR files, the environment variable `OPENCV_IO_ENABLE_OPENEXR` must be set to `1` before importing cv2.

Parameters:

| Parameter | Description                                               |
| --------- | --------------------------------------------------------- |
| file_path | The file path to which the image should be saved.         |
| image     | The image to save of shape `(height, width[, channels])`. |

Example:

```python
save_image('output.png', image)
```

### pano_to_view

```python
pano_to_view(pano, fov, orientation, view_size)
```

Creates a view image from an equirectangular panorama. The panorama is assumed to be an array of shape `(height, width[, channels])`. The FOV and orientation angles are assumed to be in DEG. The FOV is assumed to be the same in vertical and horizontal direction. The orientation is assumed to be in `(pitch, roll, yaw)` order. The order of the orientation angles, the coordinate conversions, and the spherical coordinate calculation are designed to work for right-handed ENU coordinate systems. Results may differ for other coordinate systems. Due to inaccuracies in the sampling process, it is not an exact inverse of the `view_to_pano` function.

Parameters:

| Parameter   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| pano        | The reference panorama of shape `(height, width[, channels])`. |
| fov         | The FOV [°] of the new view.                                   |
| orientation | The orientation angles [°] `(pitch, roll, yaw)` of new view.   |
| view_size   | The size of the output image of shape `(height, width)`.       |


Example:

```python
import numpy as np

pano = load_image('pano.png')

view = pano_to_view(
    pano=pano,
    fov=90,
    orientation=np.array([0, 0, 0], dtype=np.float32),
    view_size=(1024, 1024)
)

save_image('view.png', view)
```

### view_to_pano

```python
view_to_pano(view, fov, orientation, pano_size)
```

Projects a perspective view onto an equirectangular panorama. The panorama will be black except for the areas onto which the view is projected. The view is assumed to be an array of shape `(height, width[, channels])`. The FOV and orientation angles are assumed to be in DEG. The FOV is assumed to be the same in vertical and horizontal direction. The orientation is assumed to be in `(pitch, roll, yaw)` order. The order of the orientation angles, the coordinate conversions, and the spherical coordinate calculation are designed to work for right-handed ENU coordinate systems. Results may differ for other coordinate systems. Due to inaccuracies in the sampling process, it is not an exact inverse of the `pano_to_view` function.

Parameters:

| Parameter   | Description                                                        |
| ----------- | ------------------------------------------------------------------ |
| view        | The reference view of shape `(height, width[, channels])`.         |
| fov         | The FOV [°] of the reference view.                                 |
| orientation | The orientation angles [°] `(pitch, roll, yaw)` of reference view. |
| pano_size   | The size of the output image of shape `(height, width)`.           |

Example:

```python
import numpy as np

view = load_image('view.png')

pano = view_to_pano(
    view=view,
    fov=90,
    orientation=np.array([0, 0, 0], dtype=np.float32),
    pano_size=(2048, 4096)
)

save_image('pano.png', pano)
```

### view_to_view

```python
view_to_view(old_view, old_fov, old_orientation, new_fov, new_orientation, view_size)
```

Projects one perspective view into another. The new view will be black except for the areas onto which the view is projected. The view is assumed to be an array of shape `(height, width[, channels])`. The FOV and orientation angles are assumed to be in DEG. The FOV is assumed to be the same in vertical and horizontal direction. The orientation is assumed to be in `(pitch, roll, yaw)` order. The order of the orientation angles, the coordinate conversions, and the spherical coordinate calculation are designed to work for right-handed ENU coordinate systems. Results may differ for other coordinate systems. Due to inaccuracies in the sampling process, the function is not an exact inverse of itself.

Parameters:

| Parameter       | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| old_view        | The reference view of shape `(height, width[, channels])`.             |
| old_fov         | The FOV [°] of the reference view.                                     |
| old_orientation | The orientation angles [°] `(pitch, roll, yaw)` of the reference view. |
| new_fov         | The FOV [°] of the new view.                                           |
| new_orientation | The orientation angles [°] `(pitch, roll, yaw)` of the new view.       |
| view_size       | The size of the output image of shape `(height, width)`.               |

Example:

```python
import numpy as np

old_view = load_image('old_view.png')

new_view = view_to_view(
    old_view=old_view,
    old_fov=90,
    old_orientation=np.array([0, 0, 0], dtype=np.float32),
    new_fov=70,
    new_orientation=np.array([0, 0, 45], dtype=np.float32),
    view_size=(1024, 1024)
)

save_image('new_view.png', new_view)
```

### cubemaps_to_pano

```python
cubemaps_to_pano(cubemaps, pano_size)
```

Projects either 6 non-overlapping cubemaps or 18 overlapping cubemaps onto an equirectangular panorama. The cubemaps are assumed to be arrays of shape (height, width[, channels]) representing square images with a 90° horizontal and vertical FOV. For 6 cubemaps, nearest neighbor sampling is used and the views are stitched directly without applying alpha blending. For 18 cubemaps, bilinear sampling is used and the views are alpha-blended.

The 6-view layout uses orthogonal directions spaced 90° apart:
```
[NORTH, SOUTH, EAST, WEST, UP, DOWN]
```

The 18-view layout additionally includes intermediate views spaced 45° apart:
```
[NORTH, SOUTH, EAST, WEST, UP, DOWN,
 NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST,
 UPNORTH, UPSOUTH, UPEAST, UPWEST,
 DOWNNORTH, DOWNSOUTH, DOWNEAST, DOWNWEST]
```

Parameters:

| Parameter | Description                                                                 |
| --------- | --------------------------------------------------------------------------- |
| cubemaps  | The list of either 6 or 18 cubemaps of shape `(height, width[, channels])`. |
| pano_size | The size of the output image of shape `(height, width)`.                    |


Example:

```python
cubemaps = [
    load_image('cubemap_N.png'),
    load_image('cubemap_S.png'),
    load_image('cubemap_E.png'),
    load_image('cubemap_W.png'),
    load_image('cubemap_U.png'),
    load_image('cubemap_D.png')
]

pano = cubemaps_to_pano(
    cubemaps=cubemaps,
    pano_size=(2048, 4096)
)

save_image('pano.png', pano)
```


## Example Workflow

```python
import numpy as np
import os

os.environ['PPT_GPU'] = '1'

from panorama_projection_toolkit import (
    load_image,
    save_image,
    pano_to_view
)

pano = load_image('pano.png')

view = pano_to_view(
    pano=pano,
    fov=90,
    orientation=np.array([0, 0, 45], dtype=np.float32),
    view_size=(1024, 1024)
)

save_image('view.png', view)
```


## Project Structure

```text
.
├── src/
│   └── panorama_projection_toolkit/
│       ├── __init__.py
│       ├── io.py
│       ├── projections.py
│       ├── transformations.py
│       └── utils.py
└── tests/
    └── test_projections.py
```

## License

This project is licensed under the MIT License.

The MIT License is a permissive open-source license that allows anyone to use, modify, distribute, and sell this software, provided that the original copyright notice and license text are included in all copies or substantial portions of the software.

See the LICENSE file for the full license text.