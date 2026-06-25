import os

# Check if the mode has been set using the PPT_GPU environment variable
# - PPT_GPU = 0 -> CPU mode will be used
# - PPT_GPU = 1 -> GPU mode will be used
PPT_GPU = os.getenv('PPT_GPU', '1') == '1'

# Use CuPy if CUDA is available and PPT_GPU is True, otherwise NumPy and SciPy
GPU = False
if PPT_GPU:
    try:
        import cupy as xp
        from cupyx.scipy import ndimage
        GPU = True
    except ImportError:
        pass
if not GPU:
    import numpy as xp
    from scipy import ndimage


# ==========================================================
# GPU AND DATA TYPE HELPERS
# ==========================================================

def to_gpu(x: xp.ndarray) -> xp.ndarray:
    """
    Moves an array to the GPU if CuPy is available.

    Args:
        x (xp.ndarray): The array.

    Returns:
        x (xp.ndarray): The array after being moved to the GPU.
    """

    # Move the array to the GPU if CuPy is available
    if GPU:
        x = xp.asarray(x)

    return x


def to_cpu(x: xp.ndarray) -> xp.ndarray:
    """
    Moves an array to the CPU if CuPy is available.

    Args:
        x (xp.ndarray): The array.

    Returns:
        x (xp.ndarray): The array after being moved to the CPU.
    """

    # # Move the array to the CPU and release cached GPU memory blocks from the CuPy memory pool
    if GPU:
        x = xp.asnumpy(x)
        xp.get_default_memory_pool().free_all_blocks()

    return x


def restore_dtype(x: xp.ndarray, dtype: xp.dtype) -> xp.ndarray:
    """
    Converts an array to the requested dtype.

    Args:
        x (xp.ndarray): The array.
        dtype (xp.dtype): The target data type.

    Returns:
        x (xp.ndarray): The array of the requested data type.
    """

    if x.dtype == dtype:
        return x

    # Prevent overflow/underflow when converting to uint8
    if dtype == xp.uint8:
        x = xp.clip(x, 0, 255)

    # Change the data type of the array
    x = x.astype(dtype)

    return x


# ==========================================================
# GEOMETRY
# ==========================================================

def normalize_vectors(x: xp.ndarray, y: xp.ndarray, z: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Normalizes multiple 3D vectors.

    Args:
        x (xp.ndarray): The x-components of the vectors.
        y (xp.ndarray): The y-components of the vectors.
        z (xp.ndarray): The z-components of the vectors.

    Returns:
        xyz_norm (tuple[xp.ndarray, xp.ndarray, xp.ndarray]): The components of the normalized vectors.
    """

    # Calculate the length of the vectors
    length = xp.sqrt(x*x + y*y + z*z)

    # Check if the arguments are valid
    if xp.any(length <= 0):
        raise ValueError('A vector of length 0 cannot be normalized.')

    # Normalize the vectors by their length
    x_norm = x / length
    y_norm = y / length
    z_norm = z / length
    xyz_norm = (x_norm, y_norm, z_norm)

    return xyz_norm


def rotate_vectors(R: xp.ndarray, x: xp.ndarray, y: xp.ndarray,
    z: xp.ndarray) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Rotates 3D vectors by applying a 3D rotation matrix.

    Args:
        R (xp.ndarray): The rotation matrix of shape (3, 3).
        x (xp.ndarray): The x-components of the vectors.
        y (xp.ndarray): The y-components of the vectors.
        z (xp.ndarray): The z-components of the vectors.

    Returns:
        xyz_rot (tuple[xp.ndarray, xp.ndarray, xp.ndarray]): The components of the rotated vectors.
    """

    # Check if the arguments are valid
    assert x.shape == y.shape == z.shape, 'The components of the vector should be of the same shape.'
    assert R.shape == (3, 3), 'The rotation matrix should be of shape (3, 3).'
    
    # Flatten the vectors into shape (3, N) for matrix multiplication
    xyz = xp.stack((x.ravel(), y.ravel(), z.ravel()))

    # Rotate the vectors by applying the rotation matrix
    x_rot, y_rot, z_rot = xp.matmul(R, xyz).reshape(3, *x.shape)
    xyz_rot = (x_rot, y_rot, z_rot)

    return xyz_rot


def vectors_from_grid(u: xp.ndarray, v: xp.ndarray, h: int, w: int) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """
    Converts equirectangular panorama pixel coordinates into unit Cartesian direction vectors. The input coordinates
    (u, v) are interpreted as pixel indices in an equirectangular image of size (h, w), where u maps to latitude
    φ ∈ [-π/2, π/2] and v maps to longitude θ ∈ [-π, π]. The resulting vectors follow the ENU coordinate system
    convention.

    Args:
        u (xp.ndarray): The u-components of the image coordinates.
        v (xp.ndarray): The v-components of the image coordinates.
        h (int): The height of the image.
        w (int): The width of the image.

    Returns:
        xyz (tuple[xp.ndarray, xp.ndarray, xp.ndarray]): The corresponding Cartesian unit direction vectors.
    """

    # Normalize the image coordinates
    u_norm = u / (h - 1)
    v_norm = v / (w - 1)

    # Calculate the spherical coordinates from the image coordinates
    phi = (u_norm - 0.5) * xp.pi
    theta = (v_norm - 0.5) * (2 * xp.pi)

    # Calculate the sin and cos of the angles
    cos_phi = xp.cos(phi)
    sin_phi = xp.sin(phi)
    cos_theta = xp.cos(theta)
    sin_theta = xp.sin(theta)

    # Calculate the Cartesian coordinates from the spherical coordinates
    x = cos_phi * sin_theta
    y = cos_phi * cos_theta
    z = sin_phi
    xyz = (x, y, z)

    return xyz


def angles_to_R(angles):
    """
    Converts 3D Euler angles to a 3D rotation matrix. The angles are assumed to be in the order (pitch, roll, yaw),
    where pitch is the rotation around the x-axis, roll is the rotation around the y-axis, and yaw is the rotation
    around the z-axis. The axes are assumed to follow the ENU coordinate system convention. The angles are assumed
    to be given in DEG.

    Args:
        angles (xp.ndarray): The Euler angles [°] (pitch, roll, yaw).

    Returns:
        R (xp.ndarray): The corresponding rotation matrix.
    """

    # Check if the arguments are valid
    assert angles.shape == (3,), 'The angles should have exactly 3 values (pitch, roll, yaw).'

    # Convert the angles from DEG to RAD
    pitch, roll, yaw = xp.deg2rad(angles)

    # Calculate the sin and cos of the angles
    cx, sx = xp.cos(pitch), xp.sin(pitch)
    cy, sy = xp.cos(roll),  xp.sin(roll)
    cz, sz = xp.cos(yaw),   xp.sin(yaw)

    # Define values for 0 and 1 on the GPU if CuPy is available
    zero = xp.zeros((), dtype=xp.float32)
    one  = xp.ones((), dtype=xp.float32)

    # Compose the rotation matrix around the x-axis
    R_x = xp.array([
        [ one, zero, zero],
        [zero,   cx,  -sx],
        [zero,   sx,   cx]
    ], dtype=xp.float32)

    # Compose the rotation matrix around the y-axis
    R_y = xp.array([
        [  cy, zero, -sy],
        [zero,  one, zero],
        [ sy, zero,  cy]
    ], dtype=xp.float32)

    # Compose the rotation matrix around the z-axis
    R_z = xp.array([
        [  cz,  sz, zero],
        [ -sz,  cz, zero],
        [zero, zero,  one]
    ], dtype=xp.float32)

    # Combine the rotation matrices
    R = R_x @ R_y @ R_z

    return R


def get_orientations(number_of_cubemaps):
    """
    Creates a list of orientations for all cubemaps. The orientation is given as Euler angles. The angles in the order
    (pitch, roll, yaw), where pitch is the rotation around the x-axis, roll is the rotation around the y-axis, and yaw
    is the rotation around the z-axis. The axes follow the ENU coordinate system convention. The angles are given in
    DEG.

    The 6-view layout uses orthogonal directions spaced 90° apart:
        [NORTH, SOUTH, EAST, WEST, UP, DOWN]

    The 18-view layout additionally includes intermediate views spaced 45° apart:
        [NORTH, SOUTH, EAST, WEST, UP, DOWN,
         NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST,
         UPNORTH, UPSOUTH, UPEAST, UPWEST,
         DOWNNORTH, DOWNSOUTH, DOWNEAST, DOWNWEST]

    Args:
        number_of_cubemaps (int): The number of cubemaps.

    Returns:
        orientations (list[xp.ndarray]): The Euler angles [°] (pitch, roll, yaw) for each cubemap.
    """

    # Check if the arguments are valid
    assert number_of_cubemaps in [6, 18], 'The number of cubemaps should be either 6 or 18.'

    # Create the orientations for the first 6 cubemaps
    orientations = [
        xp.array([  0, 0,   0], dtype=xp.float32), # N
        xp.array([  0, 0, 180], dtype=xp.float32), # S
        xp.array([  0, 0, -90], dtype=xp.float32), # E
        xp.array([  0, 0,  90], dtype=xp.float32), # W
        xp.array([ 90, 0,   0], dtype=xp.float32), # U
        xp.array([-90, 0,   0], dtype=xp.float32), # D
    ]

    if number_of_cubemaps == 6:
        return orientations

    # Create the orientations for the remaining 12 cubemaps
    orientations += [
        xp.array([  0, 0,  -45], dtype=xp.float32), # NE
        xp.array([  0, 0,   45], dtype=xp.float32), # NW
        xp.array([  0, 0, -135], dtype=xp.float32), # SE
        xp.array([  0, 0,  135], dtype=xp.float32), # SW
        xp.array([ 45, 0,    0], dtype=xp.float32), # UN
        xp.array([ 45, 0,  180], dtype=xp.float32), # US
        xp.array([ 45, 0,  -90], dtype=xp.float32), # UE
        xp.array([ 45, 0,   90], dtype=xp.float32), # UW
        xp.array([-45, 0,    0], dtype=xp.float32), # DN
        xp.array([-45, 0,  180], dtype=xp.float32), # DS
        xp.array([-45, 0,  -90], dtype=xp.float32), # DE
        xp.array([-45, 0,   90], dtype=xp.float32)  # DW
    ]

    return orientations


# ==========================================================
# SAMPLING
# ==========================================================

def sample(image, u, v, mode='bilinear'):
    """
    Samples pixel values from an image at floating-point image coordinates. Supports nearest-neighbor and
    bilinear interpolation for both single-channel images of shape (height, width) and multi-channel images
    of shape (height, width, channels). Out-of-bounds coordinates are clipped to the valid image range.

    Args:
        image (xp.ndarray): Input image of shape (height, width[, channels]).
        u (xp.ndarray): The u-components of the image coordinates.
        v (xp.ndarray): The v-components of the image coordinates.
        mode (str): The sampling mode. Supported values are 'nearest' for nearest-neighbor interpolation and
                    'bilinear' for bilinear interpolation.

    Returns:
        samples (xp.ndarray): Sampled pixel values of shape (height, width) for single-channel images
                              and (height, width, channels) for multi-channel images.
    """

    # Get the height and width of the image
    h, w = image.shape[:2]

    # Nearest-Neigbhor Sampling
    if mode == 'nearest':

        # Get the image coordinates as int and clip them to the image size
        u = xp.clip(u.astype(int), 0, h - 1)
        v = xp.clip(v.astype(int), 0, w - 1)

        # Calculate the samples of the image at the image coordinates
        samples = image[u, v]

    # Bilinear Sampling
    else:

        # Get the image coordinates around the given coordinates as int and clip them to the image size
        u0 = xp.clip(xp.floor(u).astype(int), 0, h - 1)
        v0 = xp.clip(xp.floor(v).astype(int), 0, w - 1)
        u1 = xp.clip(u0 + 1, 0, h - 1)
        v1 = xp.clip(v0 + 1, 0, w - 1)

        # Calculate the bilinear interpolation weights
        fu = u - u0
        fv = v - v0
        w00 = (1 - fu) * (1 - fv)
        w01 = (1 - fu) * fv
        w10 = fu * (1 - fv)
        w11 = fu * fv

        # Bilinearly interpolate the pixel values
        if image.ndim == 2:
            samples = (
                image[u0, v0] * w00 +
                image[u0, v1] * w01 +
                image[u1, v0] * w10 +
                image[u1, v1] * w11
            )
        else:
            samples =  (
                image[u0, v0] * w00[..., None] +
                image[u0, v1] * w01[..., None] +
                image[u1, v0] * w10[..., None] +
                image[u1, v1] * w11[..., None]
            )

    return samples