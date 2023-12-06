import numpy as np

from services.dominant_colors_normalizer import dominant_colors_normalizer
from services.gabor_normalizer import gabor_normalizer
from services.histogram_normaliser import histograms_normalizer
from services.moments_normalizer import moments_normalizer
from services.tamura_normalizer import tamura_normalizer


def global_distance(selected_image, weights=None):
    # Assuming default weights if not provided
    if weights is None:
        weights = {
            "histogram": 1.0,
            "dominantColors": 1.0,
            "moments": 1.0,
            "gaborFilterValues": 1.0,
            "tamura": 1.0
        }

    histogram_distance = histograms_normalizer(selected_image["histogram"])
    dominant_colors_distance = dominant_colors_normalizer(selected_image["dominantColors"])
    moments_distance = moments_normalizer(selected_image["moments"])
    gabor_distance = gabor_normalizer(selected_image["gaborFilterValues"])
    tamura_distance = tamura_normalizer(selected_image["tamura"])

    # Multiply each distance by its respective weight and sum
    weighted_distances = {
        "histogram": histogram_distance * weights["histogram"],
        "dominantColors": dominant_colors_distance * weights["dominantColors"],
        "moments": moments_distance * weights["moments"],
        "gaborFilterValues": gabor_distance * weights["gaborFilterValues"],
        "tamura": tamura_distance * weights["tamura"]
    }

    global_distance = np.sum(list(weighted_distances.values()))
    return global_distance

