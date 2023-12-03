import numpy as np


def histograms_normalizer(histogram_data):
    # Normalizing each histogram in the histogram data
    normalized_histograms = [normalize_histogram(hist) for hist in histogram_data]

    # Calculate the global distance using the normalized histograms
    histogram_distance = np.mean(normalized_histograms)

    # Display the global distance
    return histogram_distance


# Function to normalize a single histogram
def normalize_histogram(hist):
    max_value = max(hist)
    if max_value != 0:
        normalized_values = [float(val) / max_value for val in hist]
        return np.mean(normalized_values)
    else:
        return hist