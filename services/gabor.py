import cv2
import numpy as np


def gabor_filter(image, orientations=None, scales=None, kernel_size=11, lambd=10, gamma=0.5, psi=0):
    """
    Calculate Gabor filter responses for a given image using specified orientations and scales.

    Args:
    - image: Input image (numpy array)
    - orientations: List of orientation angles in radians
    - scales: List of scales (sigma values)
    - kernel_size: Size of the Gabor kernel
    - sigma: Standard deviation of the Gaussian envelope
    - lambd: Wavelength of the sinusoidal factor
    - gamma: Spatial aspect ratio
    - psi: Phase offset

    Returns:
    - List of Gabor filter response values for each combination of orientations and scales
    """
    if scales is None:
        scales = [3, 5, 7]
    if orientations is None:
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gabor_values = []
        for scale in scales:
            for theta in orientations:
                gabor_kernel = cv2.getGaborKernel(
                    (kernel_size, kernel_size),
                    scale,
                    theta,
                    lambd,
                    gamma,
                    psi,
                    ktype=cv2.CV_32F
                )
                filtered_image = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
                gabor_value = np.mean(np.abs(filtered_image))
                gabor_values.append(float(gabor_value))
        return gabor_values
    except Exception as e:
        raise RuntimeError(f"Gabor filter calculation failed: {e}")
