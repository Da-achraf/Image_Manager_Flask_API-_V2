import numpy as np


def dominant_colors_normalizer(dominant_colors):
    # Normalize the dominant colors
    normalized_dominant_colors = normalize_colors(dominant_colors)

    # Calculate global distance
    dominant_colors_distance = np.mean([np.mean(color) for color in normalized_dominant_colors])
    return dominant_colors_distance


def normalize_colors(colors):
    normalized_colors = [[comp / 255.0 for comp in color] for color in colors]
    return normalized_colors