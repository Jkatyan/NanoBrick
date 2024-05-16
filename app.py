"""
Flask back-end for NanoBrick
CS 4704, Spring 2024
"""

import os
import requests
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

"""
Globals
"""

# Classes
classes = ['10928', '2465', '2780', '32009', '32014', '32034', '32054', '32062', '32072', '32073', '32140', '32184', '32248', '32270', '32271', '32291', '32348', '32449', '32498', '32523', '32526', '32556', '33299', '3647', '3648', '3649', '3673', '3706', '3713', '3737', '3749', '40490', '41239', '41678', '4185c01', '42003', '44809', '4519', '45590', '4716', '48989', '55615', '57585', '60483', '60484', '62462', '63869', '64178', '64179', '6536', '6538c', '6558', '6587', '6589', '6629', '81', '82', '83', '84', '85', '87083', '92911', '94925', '99010', '99773', 'x346']

# Object overlap threshold
OVERLAP_THRESHOLD = 0.75

# Black object threshold
BLACK_THRESHOLD = 0.05

# Initialize Flask app
app = Flask(__name__)

# Initialize inference client
custom_configuration = InferenceConfiguration(confidence_threshold=0)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ.get('RF_API_KEY')
)
CLIENT.configure(custom_configuration)

"""
Helper functions
"""
        
# Recognize image using brickognize
def recognize(image_path):
    url = "https://api.brickognize.com/predict/"
    files = {'query_image': (image_path, open(image_path, 'rb'), 'image/jpg')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        brick_info = response.json()
        return brick_info["items"]
    
# Return bounding box from prediction
def bounding_box(prediction):
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

    # Return prediction
    return {
        'name': detection_id,
        'coordinates': (left, top, right, bottom)
    }

# Remove overlaps in predictions
def remove_overlaps(predictions):
    remaining_predictions = []
    for pred in predictions:
        overlaps = False
        for other_pred in remaining_predictions:
            if overlap(pred['coordinates'], other_pred['coordinates']):
                intersection_area = calculate_intersection_area(pred['coordinates'], other_pred['coordinates'])
                pred_area = calculate_area(pred['coordinates'])
                other_pred_area = calculate_area(other_pred['coordinates'])
                # Calculate the intersection over union (IOU)
                iou = intersection_area / (pred_area + other_pred_area - intersection_area)
                # If IOU is less than OVERLAP_THRESHOLD, it's considered overlapping, so we discard the prediction
                if iou >= OVERLAP_THRESHOLD:
                    overlaps = True
                    break
        if not overlaps:
            remaining_predictions.append(pred)
    return remaining_predictions

# Calculate overlap
def overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

# Calculate intersection area
def calculate_intersection_area(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

# Calculate box area
def calculate_area(box):
    _, _, w, h = box
    return w * h

# Function to pad image with a border of given color
def pad_image(image, border_color, padding_factor):
    old_width, old_height = image.size
    new_width = old_width * padding_factor
    new_height = old_height * padding_factor
    
    padded_image = Image.new("RGB", (new_width, new_height), border_color)

    x_offset = int((new_width - old_width) / 2)
    y_offset = int((new_height - old_height) / 2)

    padded_image.paste(image, (x_offset, y_offset))
    
    return padded_image

# Detect brick color
def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define ranges for each color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([20, 255, 255])
    
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([150, 255, 255])
    
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    
    # Threshold the HSV image to get only the specified colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply a mask to filter out low brightness pixels
    v_channel = hsv[:,:,2]
    min_brightness = 75
    mask_red[v_channel < min_brightness] = 0
    mask_green[v_channel < min_brightness] = 0
    mask_blue[v_channel < min_brightness] = 0
    mask_yellow[v_channel < min_brightness] = 0
    
    # Count the number of pixels for each color
    count_red = cv2.countNonZero(mask_red)
    count_green = cv2.countNonZero(mask_green)
    count_blue = cv2.countNonZero(mask_blue)
    count_yellow = cv2.countNonZero(mask_yellow)

    # Calculate total pixels in image
    img_height, img_width, _ = image.shape
    count_total = img_height * img_width

    # Determine the dominant color
    color_counts = {'Red': count_red, 'Green': count_green, 'Blue': count_blue, 'Yellow': count_yellow}
    dominant_color = max(color_counts, key=color_counts.get)

    # Check if image is black
    if all(count / count_total < BLACK_THRESHOLD for count in color_counts.values()):
        dominant_color = None
    
    return dominant_color
    
"""
Flask app endpoint
"""

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image:
        # Unload image from request
        image_file = request.files['image']
        image_file.save(temp_image.name)
        image_path = temp_image.name
        image = Image.open(image_path)

        # Perform inference with custom model
        result_custom = CLIENT.infer(image_path, model_id="nanobrick/1")
        predictions_custom = result_custom['predictions']

        # Perform inference with RF model
        result_rf = CLIENT.infer(image_path, model_id="nanobrick/2")
        predictions_rf = result_rf['predictions']

        bricks = {}

        # Create temporary directory to store cropped images
        with tempfile.TemporaryDirectory() as cropped_output_dir:

            """
            Stage 1: Combine results from two models, remove overlaps
            """
    
            # Iterate over predictions and save to dictionary
            predictions = []

            # Custom model predictions
            for prediction in predictions_custom:
                predictions.append(bounding_box(prediction))

            # RF model predictions
            for prediction in predictions_rf:
                predictions.append(bounding_box(prediction))

            # Remove overlaps in predictions
            predictions = remove_overlaps(predictions)

            """
            Stage 2: Calculate average color, censor iteration 1 results, use custom model, remove overlaps
            """

            image_copy = image.copy()

            # Censor out predicted boxes
            draw = ImageDraw.Draw(image_copy)
            for prediction in predictions:
                draw.rectangle(prediction['coordinates'], fill=(0, 255, 0))
            image_copy = image_copy.convert("RGB")

            # Calculate average background color
            width, height = image_copy.size
            total_red = 0
            total_green = 0
            total_blue = 0
            num_valid_pixels = 0
            
            for y in range(height):
                for x in range(width):
                    r, g, b = image_copy.getpixel((x, y))
                    # Exclude pure green pixels
                    if (r, g, b) != (0, 255, 0):
                        total_red += r
                        total_green += g
                        total_blue += b
                        num_valid_pixels += 1
            
            # Fill with average background color
            avg_red = total_red / num_valid_pixels
            avg_green = total_green / num_valid_pixels
            avg_blue = total_blue / num_valid_pixels

            avg_color = (int(avg_red), int(avg_green), int(avg_blue))

            draw = ImageDraw.Draw(image_copy)
            for prediction in predictions:
                draw.rectangle(prediction['coordinates'], fill=avg_color)
            image_copy = image_copy.convert("RGB")
            image_copy_path = f"{cropped_output_dir}/censored.jpg"
            image_copy.save(image_copy_path)

            # # Perform inference with custom model on censored images
            result_custom = CLIENT.infer(image_copy_path, model_id="nanobrick/1")
            predictions_custom = result_custom['predictions']

            # Iterate over censored predictions and save to dictionary
            for prediction in predictions_custom:
                predictions.append(bounding_box(prediction))

            # Remove overlaps in predictions
            predictions = remove_overlaps(predictions)

            """
            Perform brick recognition
            """

            # Save cropped images
            for prediction in predictions:
                cropped_image = image.crop(prediction['coordinates'])
                cropped_image = cropped_image.convert("RGB")

                # Pad images
                padded_image = pad_image(cropped_image, avg_color, 3)
                padded_image.save(f"{cropped_output_dir}/{prediction['name']}.jpg")
            
            # Query brickognize
            for cropped_root, _, cropped_files in os.walk(cropped_output_dir):
                for cropped_file in cropped_files:
                    try: 
                        cropped_image_path = os.path.abspath(os.path.join(cropped_root, cropped_file))
                        items = recognize(cropped_image_path)
                        label = None
                        name = None
                        img = None

                        # Get most likely item that's within the expected classes
                        for item in items:
                            if item["id"] in classes:
                                label = item["id"]
                                name = item["name"]
                                img = item["img_url"]
                                break

                        if label is not None:
                            # Color detection if necessary
                            if label == '32523':
                                image = cv2.imread(cropped_image_path)
                                color = detect_color(image)
                                if color is None:
                                    label += f" - Black"
                                else:
                                    label += f" - {color}"

                            elif label == '43093' or label == '3749':
                                image = cv2.imread(cropped_image_path)
                                color = detect_color(image)
                                if color == 'Yellow':
                                    label = "3749 - Tan"
                                else:
                                    label = "43093 - Blue"

                            # Add to json
                            if label in bricks:
                                bricks[label]["count"] += 1
                            else:
                                bricks[label] = {"count": 1, "name": name, "image_url": img}

                    # Skip images with no result   
                    except:
                        continue
                

    # Convert bricks dictionary to JSON format
    json_data = [{"label": label, "name": data["name"], "count": data["count"], "image_url": data["image_url"]} for label, data in bricks.items()]

    # Print final prediction count for debug
    print("Final count:", len(predictions))

    return jsonify(json_data)

# Default endpoint
@app.route('/')
def home():
    return 'NanoBrick is running! Use the /predict POST endpoint to perform brick predictions.'

# Run app for debug
if __name__ == '__main__':
    # Debug: ./ngrok http --domain=penguin-gorgeous-vaguely.ngrok-free.app 5000
    app.run(debug=True, port=5000)