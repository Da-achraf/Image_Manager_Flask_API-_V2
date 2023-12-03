import numpy as np


def moments_normalizer(moments):
    # Normalize each type of moment (hue, saturation, value)
    for moment_type, moment_values in moments.items():
        # Get min and max values for the current moment type
        min_val = min(moment_values.values())
        max_val = max(moment_values.values())

        # Normalize each moment within the range [0, 1]
        for moment_key, moment_val in moment_values.items():
            moment_values[moment_key] = normalize(moment_val, min_val, max_val)

    # Calculate global distance
    moments_distance = np.mean([np.mean(list(x.values())) for x in moments.values()])
    return moments_distance


# Function to normalize values between 0 and 1
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)