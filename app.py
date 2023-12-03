import time

from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS

from services.distance import global_distance
from services.image_service import fetch_image
from services.histogram import calculate_histogram
from services.moment import calculate_moments
from services.dominantColors import find_dominant_colors
from services.gabor import gabor_filter
from services.tamura import get_tamura_features

app = Flask(__name__)
api = Api(app)

# Enable CORS for all routes
CORS(app)


class HistogramDescriptor(Resource):
    def get(self):
        start_time = time.time()
        image_url = request.args.get('image_url')

        if not image_url:
            return {"error": "Image URL not provided"}, 400

        image = fetch_image(image_url)
        if image is None:
            return {"error": "Failed to decode the image"}, 500

        histogram = calculate_histogram(image)
        end_time = time.time()
        # Calculate execution time
        execution_time = end_time - start_time
        print(f"execution time: {execution_time} seconds")
        return {"histogram": histogram}


class MomentsDescriptor(Resource):
    def get(self):
        start_time = time.time()
        image_url = request.args.get('image_url')

        if not image_url:
            return {"error": "Image URL not provided"}, 400

        image = fetch_image(image_url)
        if image is None:
            return {"error": "Failed to decode the image"}, 500

        moments = calculate_moments(image)
        end_time = time.time()
        # Calculate execution time
        execution_time = end_time - start_time
        print(f"execution time: {execution_time} seconds")
        return moments


class DominantColorsDescriptor(Resource):
    def get(self):
        start_time = time.time()
        image_url = request.args.get('image_url')
        num_colors = request.args.get('num_colors')

        if not image_url:
            return {"error": "Image URL not provided"}, 400

        image = fetch_image(image_url)
        if image is None:
            return {"error": "Failed to decode the image"}, 500

        if num_colors:
            dominant_colors = find_dominant_colors(image, int(num_colors))
        else:
            dominant_colors = find_dominant_colors(image)

        end_time = time.time()
        # Calculate execution time
        execution_time = end_time - start_time
        print(f"execution time: {execution_time} seconds")
        return {"dominantColors": dominant_colors}


class GaborFilterDescriptor(Resource):
    def get(self):
        start_time = time.time()
        image_url = request.args.get('image_url')

        if not image_url:
            return {"error": "Image URL not provided"}, 400

        image = fetch_image(image_url)
        if image is None:
            return {"error": "Failed to decode the image"}, 500

        gabor_values = gabor_filter(image)
        end_time = time.time()
        # Calculate execution time
        execution_time = end_time - start_time
        print(f"execution time: {execution_time} seconds")
        return {"gaborFilterValues": gabor_values}


class TamuraFeaturesDescriptor(Resource):
    def get(self):
        start_time = time.time()
        image_url = request.args.get('image_url')
        if not image_url:
            return {"error": "Image URL not provided"}, 400

        image = fetch_image(image_url)
        if image is None:
            return {"error": "Failed to decode the image"}, 500

        features = get_tamura_features(image)
        end_time = time.time()
        # Calculate execution time
        execution_time = end_time - start_time
        print(f"execution time: {execution_time} seconds")
        return {"tamura": features}


class SimilarImagesDescriptor(Resource):
    def post(self):
        data = request.get_json()
        selected_image = data["selectedImage"]
        other_images = data["otherImages"]
        selected_image_distance = global_distance(selected_image)
        other_images_distance = [{image["id"]: global_distance(image)} for image in other_images]

        distance_list = [
            {id_: (selected_image_distance - global_distance_image)}
            for item in other_images_distance
            for id_, global_distance_image in item.items()
        ]

        sorted_distance_list = sorted(distance_list, key=lambda x: list(x.values())[0])
        top_10_ids = [list(item.keys())[0] for item in sorted_distance_list[:10]]

        return top_10_ids


# Define a function to add resources to the API
def add_resources(api):
    api.add_resource(HistogramDescriptor, '/histogram')
    api.add_resource(MomentsDescriptor, '/moments')
    api.add_resource(DominantColorsDescriptor, '/dominant_colors')
    api.add_resource(GaborFilterDescriptor, '/gabor_filter')
    api.add_resource(TamuraFeaturesDescriptor, '/tamura_features')
    api.add_resource(SimilarImagesDescriptor, '/find_similar')


# Call the function to add resources to the API
add_resources(api)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
