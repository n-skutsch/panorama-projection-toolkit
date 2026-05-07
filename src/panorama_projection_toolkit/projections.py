from .utils import (
        xp,
        ndimage,
        to_gpu,
        to_cpu,
        restore_dtype,
        normalize_vectors,
        rotate_vectors,
        vectors_from_grid,
        angles_to_R,
        get_orientations,
        sample
    )


def pano_to_view(pano: xp.ndarray, fov: float, orientation: xp.ndarray, view_size: tuple[int, int]) -> xp.ndarray:
    """
    Creates a view image from an equirectangular panorama. The panorama is assumed to be an array of shape
    (height, width[, channels]). The FOV and orientation angles are assumed to be in DEG. The FOV is assumed to be
    the same in vertical and horizontal direction. The orientation is assumed to be in (pitch, roll, yaw) order.

    Note:
        - The order of the orientation angles, the coordinate conversions, and the spherical coordinate
          calculation are designed to work for right-handed ENU coordinate systems. Results may differ for other
          coordinate systems.
        - Due to inaccuracies in the sampling process, it is not an exact inverse of the view_to_pano function.

    Args:
        pano (xp.ndarray): The reference panorama of shape (height, width[, channels]).
        fov (float): The FOV [°] of the new view.
        orientation (xp.ndarray): The orientation angles [°] (pitch, roll, yaw) of new view.
        view_size (tuple[int, int]): The size of the output image of shape (height, width).

    Returns:
        view (xp.ndarray): The created view of shape (height, width[, channels]).
    """

    # Check if the arguments are valid
    assert pano.ndim >= 2, 'The panorama should have at least 2 dimensions.'
    assert pano.ndim <= 3, 'The panorama should have no more than 3 dimensions.'
    assert fov > 0, 'The FOV should be greater than 0.'
    assert fov < 180, 'The FOV should be smaller than 180.'
    assert orientation.ndim == 1, 'The orientation should have exactly 1 dimension.'
    assert orientation.shape == (3,), 'The orientation should have exactly 3 values (pitch, roll, yaw).'
    assert len(view_size) == 2, 'The view size should have exactly 2 values (height, width).'
    assert view_size[0] > 1, 'The height of the view should be greater than 1.'
    assert view_size[1] > 1, 'The width of the view should be greater than 1.'

    # Save the original data type of the panorama
    original_dtype = pano.dtype

    # If CuPy is available, move the panorama and orientation arrays to the GPU, and if not, keep them unchanged
    pano = to_gpu(pano)
    orientation = to_gpu(orientation)

    # Get the height and width of the view
    h, w = view_size

    # Create a meshgrid to process all pixels vectorized
    i, j = xp.meshgrid(xp.arange(h), xp.arange(w), indexing='ij')

    # Create a Cartesian projection vector for each pixel depending on the FOV and normalize it
    fov = xp.deg2rad(fov)
    x = (j / (w - 1) - 0.5) * 2 * xp.tan(fov / 2)
    y = xp.ones_like(x)
    z = (i / (h - 1) - 0.5) * 2 * xp.tan(fov / 2)
    x, y, z = normalize_vectors(x, y, z)

    # Rotate the Cartesian projection vectors according to the orientation
    R = angles_to_R(orientation)
    x, y, z = rotate_vectors(R.T, x, y, z)

    # Calculate the spherical coordinates for each rotated Cartesian view vector
    # vertical/latitude:    φ ∈ [-π/2, π/2] -> image height
    # horizontal/longitude: θ ∈ [  -π,   π] -> image width
    phi   = xp.arcsin(z)
    theta = xp.arctan2(x, y)

    # Calculate the image coordinates in the equirectangular panorama for each spherical coordinate
    u = ((phi / xp.pi) + 0.5) * (pano.shape[0] - 1)
    v = ((theta / (2 * xp.pi)) + 0.5) * (pano.shape[1] - 1)

    # Apply bilinear sampling
    view = sample(pano, u, v, mode='bilinear')

    # Move the result back to the CPU and restore the original data type
    view = to_cpu(view)
    view = restore_dtype(view, original_dtype)

    return view


def view_to_pano(view: xp.ndarray, fov: float, orientation: xp.ndarray, pano_size: tuple[int, int]) -> xp.ndarray:
    """
    Projects a perspective view onto an equirectangular panorama. The panorama will be black except for the areas
    onto which the view is projected. The view is assumed to be an array of shape (height, width[, channels]). The FOV and
    orientation angles are assumed to be in DEG. The FOV is assumed to be the same in vertical and horizontal
    direction. The orientation is assumed to be in (pitch, roll, yaw) order.

    Note:
        - The order of the orientation angles, the coordinate conversions, and the spherical coordinate
          calculation are designed to work for right-handed ENU coordinate systems. Results may differ for other
          coordinate systems.
        - Due to inaccuracies in the sampling process, it is not an exact inverse of the pano_to_view function.

    Args:
        view (xp.ndarray): The reference view of shape (height, width[, channels]).
        fov (float): The FOV [°] of the reference view. 
        orientation (xp.ndarray): The orientation angles [°] (pitch, roll, yaw) of the reference view. 
        pano_size (tuple[int, int]): The size of the output image of shape (height, width).

    Returns:
        pano (xp.ndarray): The created panorama of shape (height, width[, channels]).
    """

    # Check if the arguments are valid
    assert view.ndim >= 2, 'The view should have at least 2 dimensions.'
    assert view.ndim <= 3, 'The view should have no more than 3 dimensions.'
    assert fov > 0, 'The FOV should be greater than 0.'
    assert fov < 180, 'The FOV should be smaller than 180.'
    assert orientation.ndim == 1, 'The orientation should have exactly 1 dimension.'
    assert orientation.shape == (3,), 'The orientation should have exactly 3 values (pitch, roll, yaw).'
    assert len(pano_size) == 2, 'The panorama size should have exactly 2 values (height, width).'
    assert pano_size[0] > 1, 'The height of the panorama should be greater than 1.'
    assert pano_size[1] > 1, 'The width of the panorama should be greater than 1.'

    # Save the original data type of the view
    original_dtype = view.dtype

    # If CuPy is available, move the view and orientation arrays to the GPU, and if not, keep them unchanged
    view = to_gpu(view)
    orientation = to_gpu(orientation)

    # Get the height and width of the panorama
    h, w = pano_size

    # Create a meshgrid to process all pixels vectorized
    u_pano, v_pano = xp.meshgrid(xp.arange(h), xp.arange(w), indexing='ij')

    # Get the Cartesian projection vectors from the grid coordinates
    # vertical/latitude:    φ ∈ [-π/2, π/2] -> image height
    # horizontal/longitude: θ ∈ [  -π,   π] -> image width
    x, y, z = vectors_from_grid(u_pano, v_pano, h, w)

    # Rotate the Cartesian projection vectors according to the orientation
    R = angles_to_R(orientation)
    x, y, z = rotate_vectors(R, x, y, z)

    # Compute the pixel mask inside the camera frustum
    fov = xp.deg2rad(fov)
    valid = (
        (y > 0) &
        (xp.abs(x / y) < xp.tan(fov / 2)) &
        (xp.abs(z / y) < xp.tan(fov / 2))
    )

    # Calculate the image coordinates in the view
    u_view = ((z[valid] / y[valid]) / xp.tan(fov / 2) + 1) * 0.5 * (view.shape[0] - 1)
    v_view = ((x[valid] / y[valid]) / xp.tan(fov / 2) + 1) * 0.5 * (view.shape[1] - 1)

    # Create an empty panorama, project the view onto it, and apply bilinear sampling
    pano = xp.zeros((h, w, *view.shape[2:]), dtype=view.dtype)
    pano[valid] = sample(view, u_view, v_view, mode='bilinear')

    # Move the result back to the CPU and restore the original data type
    pano = to_cpu(pano)
    pano = restore_dtype(pano, original_dtype)

    return pano


def view_to_view(old_view: xp.ndarray, old_fov: float, old_orientation: xp.ndarray,
                 new_fov: float, new_orientation: xp.ndarray, view_size: tuple[int, int]) -> xp.ndarray:
    """
    Projects one perspective view into another. The new view will be black except for the areas onto which the view
    is projected. The view is assumed to be an array of shape (height, width[, channels]). The FOV and orientation angles
    are assumed to be in DEG. The FOV is assumed to be the same in vertical and horizontal direction. The orientation
    is assumed to be in (pitch, roll, yaw) order.

    Note:
        - The order of the orientation angles, the coordinate conversions, and the spherical coordinate
          calculation are designed to work for right-handed ENU coordinate systems. Results may differ for other
          coordinate systems.
        - Due to inaccuracies in the sampling process, the function is not an exact inverse of itself.

    Args:
        old_view (xp.ndarray): The reference view of shape (height, width[, channels]).
        old_fov (float): The FOV [°] of the reference view.
        old_orientation (xp.ndarray): The orientation angles [°] (pitch, roll, yaw) of the reference view.
        new_fov (float): The FOV [°] of the new view.
        new_orientation (xp.ndarray): The orientation angles [°] (pitch, roll, yaw) of the new view.
        view_size (tuple): The size of the output image of shape (height, width).

    Returns:
        new_view (xp.ndarray): The created view of shape (height, width[, channels]).
    """

    # Check if the arguments are valid
    assert old_view.ndim >= 2, 'The view should have at least 2 dimensions.'
    assert old_view.ndim <= 3, 'The view should have no more than 3 dimensions.'
    assert old_fov > 0, 'The FOV should be greater than 0.'
    assert old_fov < 180, 'The FOV should be smaller than 180.'
    assert old_orientation.ndim == 1, 'The orientation should have exactly 1 dimension.'
    assert old_orientation.shape == (3,), 'The orientation should have exactly 3 values (pitch, roll, yaw).'
    assert new_fov > 0, 'The FOV should be greater than 0.'
    assert new_fov < 180, 'The FOV should be smaller than 180.'
    assert new_orientation.ndim == 1, 'The orientation should have exactly 1 dimension.'
    assert new_orientation.shape == (3,), 'The orientation should have exactly 3 values (pitch, roll, yaw).'
    assert len(view_size) == 2, 'The view size should have exactly 2 values (height, width).'
    assert view_size[0] > 1, 'The height of the view should be greater than 1.'
    assert view_size[1] > 1, 'The width of the view should be greater than 1.'

    # Save the original data type of the view
    original_dtype = old_view.dtype

    # If CuPy is available, move the old view and orientation arrays to the GPU, and if not, keep them unchanged
    old_view = to_gpu(old_view)
    old_orientation = to_gpu(old_orientation)
    new_orientation = to_gpu(new_orientation)

    # Get the height and width of the new view
    h, w = view_size

    # Create a meshgrid to process all pixels vectorized
    i, j = xp.meshgrid(xp.arange(h), xp.arange(w), indexing='ij')

    # Create a Cartesian projection vector for each pixel depending on the FOV and normalize it
    new_fov_rad = xp.deg2rad(new_fov)
    x_new = (j / (w - 1) - 0.5) * 2 * xp.tan(new_fov_rad / 2)
    y_new = xp.ones_like(x_new)
    z_new = (i / (h - 1) - 0.5) * 2 * xp.tan(new_fov_rad / 2)
    x_new, y_new, z_new = normalize_vectors(x_new, y_new, z_new)

    # Rotate the Cartesian projection vectors according to the new orientation
    R_new = angles_to_R(new_orientation)
    x_new, y_new, z_new = rotate_vectors(R_new.T, x_new, y_new, z_new)

    # Rotate the Cartesian projection vectors according to the old orientation
    R_old = angles_to_R(old_orientation)
    x_old, y_old, z_old = rotate_vectors(R_old, x_new, y_new, z_new)

    # Compute the pixel mask inside the camera frustum
    old_fov_rad = xp.deg2rad(old_fov)
    valid = (
        (y_old > 0) &
        (xp.abs(x_old / y_old) < xp.tan(old_fov_rad / 2)) &
        (xp.abs(z_old / y_old) < xp.tan(old_fov_rad / 2))
    )

    # Calculate the image coordinates in the view
    u_old = ((z_old / y_old) / xp.tan(old_fov_rad / 2) + 1) * 0.5 * (old_view.shape[0] - 1)
    v_old = ((x_old / y_old) / xp.tan(old_fov_rad / 2) + 1) * 0.5 * (old_view.shape[1] - 1)

    # Create an empty view, project the old view onto it, and apply bilinear sampling
    new_view = xp.zeros((h, w, *old_view.shape[2:]), dtype=old_view.dtype)
    new_view[valid] = sample(old_view, u_old[valid], v_old[valid], mode='bilinear')

    # Move the result back to the CPU and restore the original data type
    new_view = to_cpu(new_view)
    new_view = restore_dtype(new_view, original_dtype)

    return new_view


def cubemaps_to_pano(cubemaps: list[xp.ndarray], pano_size: tuple[int, int]) -> xp.ndarray:
    """
    Projects either 6 non-overlapping cubemaps or 18 overlapping cubemaps onto an equirectangular panorama. The
    cubemaps are assumed to be arrays of shape (height, width[, channels]) representing square images with a 90° horizontal
    and vertical FOV. For 6 cubemaps, nearest neighbor sampling is used and the views are stitched directly without
    applying alpha blending. For 18 cubemaps, bilinear sampling is used and the views are alpha-blended.

    The 6-view layout uses orthogonal directions spaced 90° apart:
        [NORTH, SOUTH, EAST, WEST, UP, DOWN]

    The 18-view layout additionally includes intermediate views spaced 45° apart:
        [NORTH, SOUTH, EAST, WEST, UP, DOWN,
         NORTHEAST, NORTHWEST, SOUTHEAST, SOUTHWEST,
         UPNORTH, UPSOUTH, UPEAST, UPWEST,
         DOWNNORTH, DOWNSOUTH, DOWNEAST, DOWNWEST]
    
    Args:
        cubemaps (list[xp.ndarray]): The list of either 6 or 18 cubemaps of shape (height, width[, channels]).
        pano_size (tuple[int, int]): The size of the output image of shape (height, width).

    Returns:
        pano (xp.ndarray): The created panorama of shape (height, width[, channels]).
    """

    # Check if the arguments are valid
    assert len(cubemaps) in [6, 18], 'The list of cubemaps should contain either 6 or 18 cubemaps.'
    for cubemap in cubemaps:
        assert cubemap.ndim >= 2, 'Each cubemap should have at least 2 dimensions.'
        assert cubemap.ndim <= 3, 'Each cubemap should have no more than 3 dimensions.'
    assert len(pano_size) == 2, 'The panorama size should have exactly 2 values (height, width).'
    assert pano_size[0] > 1, 'The height of the panorama should be greater than 1.'
    assert pano_size[1] > 1, 'The width of the panorama should be greater than 1.'

    # Save the original data type of the view
    original_dtype = cubemaps[0].dtype

    # If CuPy is available, move the cubemap arrays to the GPU, and if not, keep them unchanged
    cubemaps = [to_gpu(c).astype(xp.float32) for c in cubemaps]

    # Get the height and width of the panorama
    h, w = pano_size

    # Get the orientation of each cubemap
    orientations = get_orientations(len(cubemaps))

    # Create a meshgrid to process all pixels vectorized
    u_pano, v_pano = xp.meshgrid(xp.arange(h), xp.arange(w), indexing='ij')

    # Get the Cartesian projection vectors from the grid coordinates
    # vertical/latitude:    φ ∈ [-π/2, π/2] -> image height
    # horizontal/longitude: θ ∈ [  -π,   π] -> image width
    x, y, z = vectors_from_grid(u_pano, v_pano, h, w)

    # Initialize the masks and image coordinates for all cubemaps
    masks = []
    u_all = []
    v_all = []

    # Project each cubemap onto the panorama
    for i, cube in enumerate(cubemaps):

        # Rotate the Cartesian projection vectors according to the orientation
        R = angles_to_R(orientations[i])
        rx, ry, rz = rotate_vectors(R, x, y, z)

        # Identify panorama pixels visible in this view frustum
        mask = (
            (ry > 0) &
            (xp.abs(ry) >= xp.abs(rx) - 1e-6) &
            (xp.abs(ry) >= xp.abs(rz) - 1e-6)
        )
        masks.append(mask)

        # Skip views that do not contribute to the panorama
        if not xp.any(mask):
            u_all.append(None)
            v_all.append(None)
            continue

        # Map visible rays to cubemap image coordinates
        uu = ((rz[mask] / ry[mask]) + 1) * 0.5 * (cube.shape[0] - 1)
        vv = ((rx[mask] / ry[mask]) + 1) * 0.5 * (cube.shape[1] - 1)
        uu = xp.clip(uu, 0, cube.shape[0] - 1)
        vv = xp.clip(vv, 0, cube.shape[1] - 1)
        u_all.append(uu)
        v_all.append(vv)

    # Create an empty panorama
    pano = xp.zeros((h, w, *cubemaps[0].shape[2:]), dtype=cubemaps[0].dtype)

    # 6-view stitching: nearest-neighbor sampling without blending
    if len(cubemaps) == 6:

        # Project each cubemap onto the panorama using nearest sampling
        for i, cube in enumerate(cubemaps):
            mask = masks[i]
            if not xp.any(mask):
                continue
            sampled = sample(cube, u_all[i], v_all[i], mode='nearest').astype(xp.float32)
            pano[mask] += sampled

    # 18-view stitching: bilinear sampling with alpha blending
    else:

        # Initialize the weight masks
        weights = []

        # Generate blending weights from the distance to mask boundaries
        for mask in masks:
            if xp.any(mask):
                weight = ndimage.distance_transform_edt(mask.astype(xp.float32))
                if xp.max(weight) > 0:
                    weight = weight / xp.max(weight)
            else:
                weight = xp.zeros((h, w), dtype=xp.float32)
            weights.append(weight)

        # Initialize the accumulate blending weights for normalization
        weight_sum = xp.zeros((h, w), dtype=xp.float32)

        # Project each cubemap onto the panorama using bilinear sampling
        for i, cube in enumerate(cubemaps):
            mask = masks[i]
            if not xp.any(mask):
                continue
            sampled = sample(cube, u_all[i], v_all[i], mode='bilinear').astype(xp.float32)
            weight = weights[i][mask]
            pano[mask] += sampled * weight[..., None]
            weight_sum[mask] += weight

        # Normalize each pixel by the sum of the weights
        weight_sum = xp.where(weight_sum <= 0, 1.0, weight_sum)
        if pano.ndim == 3:
            pano /= weight_sum[..., None]
        else:
            pano /= weight_sum

    # Move the result back to the CPU and restore the original data type
    pano = to_cpu(pano)
    pano = restore_dtype(pano, original_dtype)

    return pano