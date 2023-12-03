import cv2
import numpy as np


def calculate_histogram(image):
    # Calculate the histogram for each channel (Hue, Saturation, Value) directly
    histograms = [
        cv2.calcHist([image], [i], None, [256], [0, 256]).flatten().tolist()
        for i in range(3)
    ]

    return histograms
