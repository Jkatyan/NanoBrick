"""
Flask back-end for NanoBrick
CS 4704, Spring 2024
"""

import os
import cv2
import numpy as np
import argparse
import requests
import tempfile
from PIL import Image
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Initialize Flask app
app = Flask(__name__)

# Initialize inference client
custom_configuration = InferenceConfiguration(confidence_threshold=0)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ.get('RF_API_KEY')
)
CLIENT.configure(custom_configuration)

# Constants
EXPANDED_PIXELS = 0
MIN_AREA_THRESHOLD = 10000

"""
Image processing helper functions
"""

# Sobel edge detection
def sobel_edge_detection(image, threshold=100):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    edges = np.zeros_like(gradient_magnitude)
    edges[gradient_magnitude > threshold] = 255
    return edges


# Gaussian blur
def preprocess(image, power):
    blurred = cv2.GaussianBlur(image, (power, power), 0)
    return blurred


# Contour detection
def find_external_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Convert to black and white
def black_and_white(image_path, output_path):
    image = cv2.imread(image_path) 
    blur = preprocess(image, 41)
    edges = sobel_edge_detection(blur, threshold=30)
    contours = find_external_contours(edges)
    contour_image = np.zeros_like(edges)
    cv2.drawContours(contour_image, contours, -1, (255, 0,), thickness=cv2.FILLED)
    cv2.imwrite(output_path, contour_image)


# Expand bounding boxes
def expand_bounding_box(box, expand_pixels, image_width, image_height):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - expand_pixels)
    y1 = max(0, y1 - expand_pixels)
    x2 = min(image_width, x2 + expand_pixels)
    y2 = min(image_height, y2 + expand_pixels)
    return x1, y1, x2, y2


# Create bounding boxes and save cropped images
def create_bounding_boxes(black_and_white, original, output_path):
    image = cv2.imread(black_and_white)
    blur = preprocess(image, 41)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    bounding_boxes = []
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_box = expand_bounding_box((x, y, x + w, y + h), EXPANDED_PIXELS, image.shape[1], image.shape[0])
            bounding_boxes.append(bounding_box)

    original = cv2.imread(original)
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box
        cropped_image = original[y1:y2, x1:x2]

        # Apply padding
        height, width, _ = cropped_image.shape
        new_height = height * 2
        new_width = width * 2
        padded_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        padded_image[y_offset:y_offset+height, x_offset:x_offset+width] = cropped_image

        cv2.imwrite(os.path.join(output_path, f'cropped_{i}.jpg'), padded_image)

"""
Pipeline helper functions
"""

# Extract images from input file and store cropped images in output dir
def extract_images_pipeline(input_file, output_directory):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            black_and_white_path = tmp_file.name
            black_and_white(input_file, black_and_white_path)
            create_bounding_boxes(black_and_white_path, input_file, output_directory)
        os.remove(black_and_white_path)
        

# Recognize image using brickognize
def recognize(image_path):
    url = "https://api.brickognize.com/predict/"
    files = {'query_image': (image_path, open(image_path, 'rb'), 'image/jpg')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        brick_info = response.json()
        label = brick_info["items"][0]["id"]
        name = brick_info["items"][0]["name"]
        img = brick_info["items"][0]["img_url"]

        return label, name, img


# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Brick segmentation script.")
    parser.add_argument("input_file", type=str, help="Path to the image file")
    parser.add_argument("output_file", type=str, help="Path to the output file")
    return parser.parse_args()

"""
Flask app endpoint
"""

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image:
        image_file = request.files['image']
        image_file.save(temp_image.name)

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Extract images
        extract_images_pipeline(temp_image.name, temp_output_dir)

        bricks = {}
        for root, _, files in os.walk(temp_output_dir):
            for file in files:
                # Get image information
                image_path = os.path.abspath(os.path.join(root, file))
                image = Image.open(image_path)

                # Predict boxes
                result = CLIENT.infer(image_path, model_id="nanobrick/1")
                predictions = result['predictions']

                # More than one detected brick, perform segmentation with hosted model
                if len(predictions) > 1:
                    # Create temporary directory to store cropped images
                    with tempfile.TemporaryDirectory() as cropped_output_dir:
                
                        # Iterate over predictions
                        for prediction in predictions:
                            x = int(prediction['x'])
                            y = int(prediction['y'])
                            width = int(prediction['width'])
                            height = int(prediction['height'])
                            detection_id = prediction['detection_id']
                            
                            # Crop the image based on bounding box
                            left = x - width // 2
                            top = y - height // 2
                            right = x + width // 2
                            bottom = y + height // 2
                            cropped_image = image.crop((left, top, right, bottom))
                            
                            # Save cropped image
                            cropped_image.save(f"{cropped_output_dir}/{detection_id}.jpg")
                        
                        # Query brickognize
                        for cropped_root, _, cropped_files in os.walk(cropped_output_dir):
                            for cropped_file in cropped_files:
                                try: 
                                    cropped_image_path = os.path.abspath(os.path.join(cropped_root, cropped_file))
                                    label, name, img = recognize(cropped_image_path)
                                    if label in bricks:
                                        bricks[label]["count"] += 1
                                    else:
                                        bricks[label] = {"count": 1, "name": name, "image_url": img}

                                # Skip images with no result   
                                except:
                                    continue
                
                # One or less detected bricks, query brickognize directly
                else:
                    try: 
                        label, name, img = recognize(image_path)
                        if label in bricks:
                            bricks[label]["count"] += 1
                        else:
                            bricks[label] = {"count": 1, "name": name, "image_url": img}

                    # Skip images with no result   
                    except:
                        continue

    # Convert bricks dictionary to JSON format
    json_data = [{"label": label, "name": data["name"], "count": data["count"], "image_url": data["image_url"]} for label, data in bricks.items()]

    return jsonify(json_data)


# Default endpoint
@app.route('/')
def home():
    return 'NanoBrick is running! Use the /predict POST endpoint to perform brick predictions.'


# Run app for debug
if __name__ == '__main__':
    app.run(debug=True)