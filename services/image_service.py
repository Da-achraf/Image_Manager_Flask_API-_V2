import cv2
import numpy as np
import requests


# Helper function to fetch and decode the image from the URL
def fetch_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_bytes = response.content
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    return None
