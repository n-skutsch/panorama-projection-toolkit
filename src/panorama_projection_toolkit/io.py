import cv2
import numpy as np
import warnings

from pathlib import Path


def load_image(file_path: str|Path) -> np.ndarray:
    """
    Loads an image from the specified file path. EXR files are typically loaded as floating-point arrays
    (e.g., float32), whereas all other file types are typically loaded as uint8 arrays. The image is returned
    as a NumPy array of shape (height, width) for single-channel images and (height, width, channels) for
    multi-channel images. For color images, the color channels are in BGR order.

    Note:
        - Non-EXR grayscale images are also loaded as 3-channel BGR images.
        - In order to load images from EXR files, the environment variable OPENCV_IO_ENABLE_OPENEXR
          must be set to 1 before importing cv2.

    Args:
        file_path (str|Path): The file path at which the image is located. If given as a str, it is converted
                              internally to a Path object.

    Returns:
        image (np.ndarray): The loaded image of shape (height, width) for single-channel images
                            and (height, width, channels) for multi-channel images.
    """

    # Convert the file path to a Path object if given as a str
    file_path = Path(file_path)

    # Check if the file exists
    if not file_path.exists():
        raise FileNotFoundError('The input file "{0}" could not be found.'.format(file_path))

    # Load EXR images preserving the original dtype and channels
    if file_path.suffix.lower() == '.exr':
        image = cv2.imread(str(file_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    # Load all other images as 8-bit BGR images
    else:
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

    # Check if the image could be loaded
    if image is None:
        raise ValueError('Failed to load image "{0}". Unsupported or corrupted file.'.format(file_path))

    return image


def save_image(file_path: str|Path, image: np.ndarray) -> None:
    """
    Saves an image to the specified file path. The image is expected to be a NumPy array of shape (height, width) for
    single-channel images and (height, width, channels) with color channels in BGR order for multi-channel images. If
    the file extension is either JPG/JPEG or PNG and the image is not an uint8 array, it is converted to an uint8 array
    before saving it. If the file extension is EXR and the image is not a float32 array, it is converted to a float32
    array before saving it. These conversions may lead to unexpected results.

    Note:
        - The pixel values of float images are expected to fall within the range of [0, 1].
        - The pixel values of uint8 images are expected to fall within the range of [0, 255].
        - In order to save images to EXR files, the environment variable OPENCV_IO_ENABLE_OPENEXR
          must be set to 1 before importing cv2.

    Args:
        file_path (str|Path): The file path to which the image should be saved. If given as a str, it is converted
                              internally to a Path object.
        image (np.ndarray): The image to save of shape (height, width) for single-channel images
                            and (height, width, channels) for multi-channel images.

    Returns:
        None
    """

    # Convert the file path to a Path object if given as a str
    file_path = Path(file_path)

    # Check if the file extension corresponds to the data type of the image
    file_ext = file_path.suffix.lower()
    if file_ext in ['.jpg', '.jpeg', '.png'] and image.dtype != np.uint8:
        image = np.uint8(image * 255)
        warnings.warn('Saving an image with a data type other than uint8 as JPG/JPEG/PNG may lead to unexpected results.',
                      UserWarning)
    if file_ext in ['.jpg', '.jpeg'] and image.ndim == 3 and image.shape[2] == 4:
        warnings.warn('Saving an image with alpha channel as JPG/JPEG may lead to unexpected results. ' + \
                      'The alpha channel will be discarded.', UserWarning)
    if file_ext in ['.exr'] and image.dtype != np.float32:
        image = np.float32(image / 255)
        warnings.warn('Saving an image with a data type other than float32 as EXR may lead to unexpected results.',
                      UserWarning)

    # Save the image
    if file_ext in ['.jpg', '.jpeg']:
        successful_write = cv2.imwrite(str(file_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        successful_write = cv2.imwrite(str(file_path), image)

    # Check if the image has been saved
    if not successful_write:
        raise RuntimeError('Failed to save image "{0}".'.format(file_path))

    return