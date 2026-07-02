import cv2
import numpy as np
import os

# Set the mode to either CPU (0) or GPU (1)
os.environ['PPT_GPU'] = '1'

from panorama_projection_toolkit import (
        load_image,
        save_image,
        pano_to_view,
        view_to_pano,
        view_to_view,
        cubemaps_to_pano,
        pano_skyline_to_elevations,
        pano_elevations_to_skyline,
        view_skyline_to_elevations,
        view_elevations_to_skyline
    )


def put_centered_text(image: np.ndarray, text: str) -> np.ndarray:
    """
    Draws text centered in the image. The text is scaled to be as large as possible
    while keeping a 20% border on each side in x and y.

    Args:
        image (np.ndarray): The input image of shape (height, width[, 3]).
        text (str): The text to render.

    Returns:
        text_image (np.ndarray): The image with the text on it.
    """

    # Copy the original image
    text_image = image.copy()
    H, W = text_image.shape[:2]

    # Define the allowed drawing region as 60% of the image in each direction
    max_width  = int(W * 0.6)
    max_height = int(H * 0.6)

    # Define the font and scale the thickness with the image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(min(H, W) * 0.05))

    # Find the optimal scale for the text
    low_bound, high_bound = 0.1, 100.0
    best_scale = low_bound
    for _ in range(30):
        mid = (low_bound + high_bound) / 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, mid, thickness)
        if text_width <= max_width and text_height <= max_height:
            best_scale = mid
            low_bound = mid
        else:
            high_bound = mid

    # Get the final text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, best_scale, thickness)

    # Calculate the center position
    x = (W - text_width)  // 2
    y = (H + text_height) // 2

    # Draw the text onto the image
    cv2.putText(text_image, text, (x, y), font, best_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return text_image


def main():

    # Create test panorama image
    y = np.linspace(0, 255, 2000)[:, None]
    x = np.linspace(0, 255, 4000)[None, :]
    gradient = (x + y) / 2
    input_pano = np.zeros((2000, 4000, 3), dtype=np.uint8)
    input_pano[..., 2] = gradient
    input_pano[..., 1] = 255 - gradient

    # Create test cubemap images
    input_cube_N  = np.tile([255, 0, 0], (1000, 1000, 1)).astype(np.uint8)
    input_cube_S  = np.tile([255, 0, 0], (1000, 1000, 1)).astype(np.uint8)
    input_cube_E  = np.tile([0, 0, 255], (1000, 1000, 1)).astype(np.uint8)
    input_cube_W  = np.tile([0, 0, 255], (1000, 1000, 1)).astype(np.uint8)
    input_cube_U  = np.tile([0, 255, 0], (1000, 1000, 1)).astype(np.uint8)
    input_cube_D  = np.tile([0, 255, 0], (1000, 1000, 1)).astype(np.uint8)
    input_cube_NE = (input_cube_N + input_cube_E) // 2
    input_cube_NW = (input_cube_N + input_cube_W) // 2
    input_cube_SE = (input_cube_S + input_cube_E) // 2
    input_cube_SW = (input_cube_S + input_cube_W) // 2
    input_cube_UN = (input_cube_U + input_cube_N) // 2
    input_cube_US = (input_cube_U + input_cube_S) // 2
    input_cube_UE = (input_cube_U + input_cube_E) // 2
    input_cube_UW = (input_cube_U + input_cube_W) // 2
    input_cube_DN = (input_cube_D + input_cube_N) // 2
    input_cube_DN = (input_cube_D + input_cube_S) // 2
    input_cube_DE = (input_cube_D + input_cube_E) // 2
    input_cube_DW = (input_cube_D + input_cube_W) // 2
    input_cube_N  = put_centered_text(input_cube_N, 'N')
    input_cube_S  = put_centered_text(input_cube_S, 'S')
    input_cube_E  = put_centered_text(input_cube_E, 'E')
    input_cube_W  = put_centered_text(input_cube_W, 'W')
    input_cube_U  = put_centered_text(input_cube_U, 'U')
    input_cube_D  = put_centered_text(input_cube_D, 'D')
    input_cube_NE = put_centered_text(input_cube_NE, 'NE')
    input_cube_NW = put_centered_text(input_cube_NW, 'NW')
    input_cube_SE = put_centered_text(input_cube_SE, 'SE')
    input_cube_SW = put_centered_text(input_cube_SW, 'SW')
    input_cube_UN = put_centered_text(input_cube_UN, 'UN')
    input_cube_US = put_centered_text(input_cube_US, 'US')
    input_cube_UE = put_centered_text(input_cube_UE, 'UE')
    input_cube_UW = put_centered_text(input_cube_UW, 'UW')
    input_cube_DN = put_centered_text(input_cube_DN, 'DN')
    input_cube_DN = put_centered_text(input_cube_DN, 'DN')
    input_cube_DE = put_centered_text(input_cube_DE, 'DE')
    input_cube_DW = put_centered_text(input_cube_DW, 'DW')
    cubemaps_no_blend = [
            input_cube_N, input_cube_S, input_cube_E, input_cube_W, input_cube_U, input_cube_D
        ]
    cubemaps_blend = [
            input_cube_N, input_cube_S, input_cube_E, input_cube_W, input_cube_U, input_cube_D,
            input_cube_NE, input_cube_NW, input_cube_SE, input_cube_SW,
            input_cube_UN, input_cube_US, input_cube_UE, input_cube_UW,
            input_cube_DN, input_cube_DN, input_cube_DE, input_cube_DW
        ]

    # Test pano_to_view
    save_image('test_pano_to_view.png', pano_to_view(input_pano, 90, np.array([10, 0, 90]), (1000, 1000)))

    # Test view_to_pano
    save_image('test_view_to_pano_N.png', view_to_pano(input_cube_N, 90, np.array([ 0, 0,   0]), (2000, 4000)))
    save_image('test_view_to_pano_E.png', view_to_pano(input_cube_E, 90, np.array([ 0, 0, -90]), (2000, 4000)))
    save_image('test_view_to_pano_U.png', view_to_pano(input_cube_U, 90, np.array([90, 0,   0]), (2000, 4000)))

    # Test view_to_view
    save_image('test_view_to_view_N.png', view_to_view(input_cube_N, 90, np.array([ 0, 0,   0]),
                                                       90, np.array([45, 0, -45]), (1000, 1000)))
    save_image('test_view_to_view_E.png', view_to_view(input_cube_E, 90, np.array([ 0, 0, -90]),
                                                       90, np.array([45, 0, -45]), (1000, 1000)))
    save_image('test_view_to_view_U.png', view_to_view(input_cube_U, 90, np.array([90, 0,   0]),
                                                       90, np.array([45, 0, -45]), (1000, 1000)))

    # Test cubemaps_to_pano
    save_image('test_cubemaps_to_pano_no_blend.png', cubemaps_to_pano(cubemaps_no_blend, (2000, 4000)))
    save_image('test_cubemaps_to_pano_blend.png', cubemaps_to_pano(cubemaps_blend, (2000, 4000)))


if __name__ == '__main__':
    main()