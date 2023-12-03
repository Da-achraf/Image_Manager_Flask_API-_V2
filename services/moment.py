import cv2
import numpy as np


def calculate_moments(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the moments directly
    moments = {}
    for channel_name, channel in zip(['Hue', 'Saturation', 'Value'], cv2.split(hsv_image)):
        moments[channel_name + '_Mean'] = np.mean(channel)
        moments[channel_name + '_StdDev'] = np.std(channel)
        moments[channel_name + '_Skewness'] = np.power(
            cv2.moments(channel)['mu20'] * cv2.moments(channel)['mu02'], 1 / 4
        )

    formatted_moments = format_moments(moments)
    return formatted_moments


def format_moments(moments):
    return {
        "moments": {
            "hue": {
                "mean": moments.get("Hue_Mean"),
                "stdDeviation": moments.get("Hue_StdDev"),
                "skewness": moments.get("Hue_Skewness")
            },
            "saturation": {
                "mean": moments.get("Saturation_Mean"),
                "stdDeviation": moments.get("Saturation_StdDev"),
                "skewness": moments.get("Saturation_Skewness")
            },
            "value": {
                "mean": moments.get("Value_Mean"),
                "stdDeviation": moments.get("Value_StdDev"),
                "skewness": moments.get("Value_Skewness")
            }
        }
    }