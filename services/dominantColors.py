import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

def find_dominant_colors(image, k=10, downsample_ratio=0.1):
    # Downsample the image
    small_image = cv2.resize(image, None, fx=downsample_ratio, fy=downsample_ratio)

    # Reshape the downsampled image to a 2D array of pixels
    pixels = np.float32(small_image.reshape(-1, 3))

    # Apply MiniBatchKMeans from scikit-learn
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1000, max_iter=100, n_init=3)
    kmeans.fit(pixels)

    # Get centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Get the counts of labels to find the dominant colors
    label_counts = Counter(labels)

    # Sort label counts to get top k dominant colors
    top_labels = sorted(label_counts, key=label_counts.get, reverse=True)[:k]

    # Get RGB values of the top k dominant colors
    dominant_colors = [centroids[label] for label in top_labels]
    dominant_colors = [[int(value) for value in color] for color in dominant_colors]

    return dominant_colors
