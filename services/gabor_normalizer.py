import numpy as np


def gabor_normalizer(gabor_values):
    # Calculate the minimum and maximum values
    min_value = min(gabor_values)
    max_value = max(gabor_values)

    # Normalize the values between 0 and 1
    normalized_values = [(value - min_value) / (max_value - min_value) for value in gabor_values]

    # Calculate the global distance
    gabor_distance = np.mean(normalized_values)
    return gabor_distance

