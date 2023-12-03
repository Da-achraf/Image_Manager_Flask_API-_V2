import numpy as np

from services.dominant_colors_normalizer import dominant_colors_normalizer
from services.gabor_normalizer import gabor_normalizer
from services.histogram_normaliser import histograms_normalizer
from services.moments_normalizer import moments_normalizer
from services.tamura_normalizer import tamura_normalizer


def global_distance(selected_image):
    histogram_distance = histograms_normalizer(selected_image["histogram"])
    dominant_colors_distance = dominant_colors_normalizer(selected_image["dominantsColors"])
    moments_distance = moments_normalizer(selected_image["moments"])
    gabor_distance = gabor_normalizer(selected_image["gaborFilterValues"])
    tamura_distance = tamura_normalizer(selected_image["tamura"])
    print(histogram_distance)
    print(dominant_colors_distance)
    print(moments_distance)
    print(gabor_distance)
    print(tamura_distance, "\n")
    global_distance = np.mean([histogram_distance, dominant_colors_distance, moments_distance, gabor_distance, tamura_distance])
    return global_distance

