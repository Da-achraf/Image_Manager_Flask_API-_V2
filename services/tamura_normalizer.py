import numpy as np


def tamura_normalizer(tamura_features):
    # Calculate the minimum and maximum values
    min_value = min(tamura_features.values())
    max_value = max(tamura_features.values())

    # Normalize each feature using Min-Max normalization
    normalized_tamura_features = [
        normalize(value, min_value, max_value) for value in list(tamura_features.values())
    ]

    # Calculate the global distance
    tamura_distance = np.mean([value for value in normalized_tamura_features])

    # Display the global distance
    return tamura_distance


# Function to normalize values between 0 and 1
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)
