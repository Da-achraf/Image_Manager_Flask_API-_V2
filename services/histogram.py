import cv2


def calculate_histogram(image):
    # Split the image into its color channels
    chans = cv2.split(image)

    # Define colors and initialize an empty list for histograms
    colors = ("b", "g", "r")
    histograms = []

    # Calculate histogram for each channel and append to the histograms list
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten().tolist()
        histograms.append(hist)

    return histograms

