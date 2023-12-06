import cv2
import numpy as np


def get_tamura_features(image):
    """
    Function to get Tamura features of an image
    :param image: OpenCV grayscale image ndarray like
    :return: Returns a dictionary of features (coarseness, contrast, directionality...etc.).
    """
    features = dict()
    features["coarseness"] = float(get_coarseness_tamura(image))
    features["contrast"] = float(get_contrast(image))
    features["directionality"] = float(get_directionality(image))
    features["regularity"] = float(calculate_regularity(image))
    features["roughness"] = float(calculate_roughness(image))
    features["linelikeness"] = float(get_linelikeness(image))

    return features


def get_coarseness_tamura(image):

    # Resize the image if it's larger than 200x200
    # image = cv2.resize(image, (min(200, image.shape[1]), min(200, image.shape[0])))
    H, W = image.shape[:2]
    Ei = []
    SBest = np.zeros((H, W))

    for k in range(1, 7):
        Ai = np.zeros((H, W))
        Ei_h = np.zeros((H, W))
        Ei_v = np.zeros((H, W))

        # Calculate Ai
        for h in range(2 ** (k - 1) + 1, H - (k - 1)):
            for w in range(2 ** (k - 1) + 1, W - (k - 1)):
                Ai[h, w] = np.sum(image[h - (2 ** (k - 1) - 1): h + (2 ** (k - 1) - 1) - 1,
                                  w - (2 ** (k - 1) - 1): w + (2 ** (k - 1) - 1) - 1])

        # Calculate Ei_h and Ei_v
        for h in range(2 ** (k - 1) + 1, H - k):
            for w in range(2 ** (k - 1) + 1, W - k):
                try:
                    Ei_h[h, w] = Ai[h + (2 ** (k - 1) - 1), w] - Ai[h - (2 ** (k - 1) - 1), w]
                    Ei_v[h, w] = Ai[h, w + (2 ** (k - 1) - 1)] - Ai[h, w - (2 ** (k - 1) - 1)]
                except IndexError:
                    pass

        # Normalize Ei_h and Ei_v
        Ei_h /= 2 ** (2 * k)
        Ei_v /= 2 ** (2 * k)

        Ei.append(Ei_h)
        Ei.append(Ei_v)

    Ei = np.array(Ei)
    for h in range(H):
        for w in range(W):
            maxv_index = np.argmax(Ei[:, h, w])
            k_temp = (maxv_index + 1) // 2
            SBest[h, w] = 2 ** k_temp

    coarseness = np.sum(SBest) / (H * W)
    return coarseness


def get_contrast(image):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    count_probs = hist / np.sum(hist)

    mean = np.mean(count_probs)
    moment_2nd = np.sum((np.arange(len(count_probs)) - mean) ** 2 * count_probs)
    moment_4th = np.sum((np.arange(len(count_probs)) - mean) ** 4 * count_probs)

    std = np.sqrt(moment_2nd)
    alfa4 = moment_4th / (std ** 4)

    n = 0.25
    tamura_contrast = std / (alfa4 ** n)
    return tamura_contrast


def get_directionality(image, threshold=12):
    # Sobel derivatives to compute gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and angle
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    angle = np.arctan2(sobely, sobelx)

    # Quantization into 9 bins
    bins = np.linspace(-np.pi, np.pi, 10)
    quantized_angles = np.digitize(angle, bins) - 1

    # Thresholded gradients
    thresholded_magnitude = (magnitude > threshold) * 1

    # Generate histograms for quantized angles based on thresholded gradients
    dir_vector, _ = np.histogram(quantized_angles[thresholded_magnitude.astype(bool)], bins=9, range=(0, 9))

    # Normalize histogram
    dir_vector = dir_vector / np.mean(dir_vector)

    # Calculate directionality feature
    n = 9  # Align with the number of bins in dir_vector
    hd_max_index = np.argmax(dir_vector)
    fdir = np.sum((np.arange(n) - hd_max_index) ** 2 * dir_vector)

    return fdir


def get_linelikeness(image):
    # Convert the image to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Sobel derivatives to compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and angle
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    angle = np.arctan2(sobely, sobelx)

    # Quantization into 9 bins
    bins = np.linspace(-np.pi, np.pi, 10)
    quantized_angles = np.digitize(angle, bins) - 1

    # Thresholded gradients
    threshold = np.mean(magnitude)
    thresholded_magnitude = (magnitude > threshold) * 1

    # Generate histograms for quantized angles based on thresholded gradients
    hist, _ = np.histogram(quantized_angles[thresholded_magnitude.astype(bool)], bins=9, range=(0, 9))

    # Normalize histogram
    hist = hist / np.sum(hist)

    # Calculate Linelikeness feature
    n = 9  # Number of bins
    hd_max_index = np.argmax(hist)
    linelikeness = np.sum((np.arange(n) - hd_max_index) ** 2 * hist)

    return linelikeness


def calculate_lbp(image):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    radius = 1
    num_points = 8 * radius

    # Compute LBP using OLBP (for OpenCV 3.x) or ORB (for OpenCV 4.x)
    lbp = cv2.OLBP(image, radius, num_points) if cv2.__version__.startswith('3') else cv2.ORB_create().detectAndCompute(image, None)[1]
    return lbp


def calculate_regularity(image):
    lbp = calculate_lbp(image)
    hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0, 256])
    hist /= np.sum(hist)
    regularity = -np.sum(hist * np.log2(hist + 1e-10))
    return regularity


def calculate_roughness(image):
    # Convert the image to grayscale if it's a color image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scales = [3, 5, 7]  # You can adjust these scales as needed

    roughness_values = []
    for scale in scales:
        blurred = cv2.GaussianBlur(image, (scale, scale), 0)
        diff = cv2.absdiff(image, blurred)
        roughness = np.var(diff)
        roughness_values.append(roughness)

    tamura_roughness = np.mean(roughness_values)
    return tamura_roughness

