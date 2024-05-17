"""
Flask back-end for NanoBrick
CS 4704, Spring 2024
"""

import os
import requests
import tempfile
import cv2
import numpy as np
import threading
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

"""
Globals
"""

# Kit groups
group_1 = ['32449', '33299', '41678', '99773', '87082', '48989', '55615']
group_2 = ['6538c', '6536', '57585', '44809', '63869', '32291', '42003', '32184', '32014', '32013', '32034']
group_3 = ['32523', '60483', '45590', 'x346']
group_4 = ['32062', '4519', '3705', '32073', '3706', '44294', '3707', '60485', '3737', '3708']
group_5 = ['6558', '32556']
group_6 = ['2780', '3673']
group_7 = ['32054', '62462', '87083', '55013', '6587']
group_8 = ['3749', '43093']
group_9 = ['4265c', '3713']
group_10 = ['32316', '32524', '40490', '32525', '41239', '32278']
group_11 = ['32140', '60484', '32526', '32348', '32271', '6629', '32009']
group_12 = ['64179', '64178', '92911']
group_13 = ['10928', '6589', '94925', '3648', '3649', '32270', '32269', '32498', '99010', '4716', '32072', '4185c01']

# All classes
groups = [group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10, group_11, group_12, group_13]
classes = group_1 + group_2 + group_3 + group_4 + group_5 + group_6 + group_7 + group_8 + group_9 + group_10 + group_11 + group_12 + group_13

# Object overlap threshold
DEFAULT_OVERLAP_THRESHOLD = 0.7
OVERLAP_THRESHOLD_1 = 0.75
OVERLAP_THRESHOLD_2 = 0.85
OVERLAP_THRESHOLD_3 = 0.6
OVERLAP_THRESHOLD_4 = 0.98
OVERLAP_THRESHOLD_5 = 0.7
OVERLAP_THRESHOLD_6 = 0.9
OVERLAP_THRESHOLD_7 = 0.9
OVERLAP_THRESHOLD_8 = 0.7
OVERLAP_THRESHOLD_9 = 0.7
OVERLAP_THRESHOLD_10 = 0.93
OVERLAP_THRESHOLD_11 = 0.85
OVERLAP_THRESHOLD_12 = 0.6
OVERLAP_THRESHOLD_13 = 0.85
overlap_thresholds = [OVERLAP_THRESHOLD_1, OVERLAP_THRESHOLD_2, OVERLAP_THRESHOLD_3, OVERLAP_THRESHOLD_4, OVERLAP_THRESHOLD_5, OVERLAP_THRESHOLD_6, OVERLAP_THRESHOLD_7, OVERLAP_THRESHOLD_8, OVERLAP_THRESHOLD_9, OVERLAP_THRESHOLD_10, OVERLAP_THRESHOLD_11, OVERLAP_THRESHOLD_12, OVERLAP_THRESHOLD_13]

# Black object threshold
BLACK_THRESHOLD = 0.05

# Create global lock
lock = threading.Lock()

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
def remove_overlaps(predictions, threshold):
    remaining_predictions = []
    for pred in predictions:
        overlaps = False
        for other_pred in remaining_predictions:
            if overlap(pred['coordinates'], other_pred['coordinates']):
                intersection_area = calculate_intersection_area(pred['coordinates'], other_pred['coordinates'])
                pred_area = calculate_area(pred['coordinates'])
                other_pred_area = calculate_area(other_pred['coordinates'])
                iou = intersection_area / (pred_area + other_pred_area - intersection_area)
                # If IOU is less than threshold, it's considered overlapping, so we discard the prediction
                if iou >= threshold:
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
    
    lower_blue = np.array([70, 50, 50])
    upper_blue = np.array([170, 255, 255])
    
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply a mask to filter out low brightness pixels
    v_channel = hsv[:,:,2]
    min_brightness = 100
    mask_blue[v_channel < min_brightness] = 0
    mask_yellow[v_channel < min_brightness] = 0
    
    # Count the number of pixels for each color
    count_blue = cv2.countNonZero(mask_blue)
    count_yellow = cv2.countNonZero(mask_yellow)

    # Calculate total pixels in image
    img_height, img_width, _ = image.shape
    count_total = img_height * img_width

    # Determine the dominant color
    color_counts = {'Blue': count_blue, 'Yellow': count_yellow}
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
    with lock:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image:
            # Unload image from request
            image_file = request.files['image']
            image_file.save(temp_image.name)
            image_path = temp_image.name
            image = Image.open(image_path)

            info = f"Starting Prediction, ID: {temp_image.name}"
            print('=' * len(info))
            print(f"{info}")

            # Perform inference with custom model
            result_custom = CLIENT.infer(image_path, model_id="nanobrick/1")
            predictions_custom = result_custom['predictions']

            # Perform inference with RF model
            result_rf = CLIENT.infer(image_path, model_id="nanobrick/2")
            predictions_rf = result_rf['predictions']

            bricks = {}
            overlap_threshold = DEFAULT_OVERLAP_THRESHOLD
            selected_group = classes

            """
            Perform predictions with default overlap threshold
            """

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
                predictions = remove_overlaps(predictions, overlap_threshold)

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

                    """
                    Query brickognize, determine correct kit group
                    """

                    initial_bricks = []
                    for cropped_file in cropped_files:
                        try: 
                            cropped_image_path = os.path.abspath(os.path.join(cropped_root, cropped_file))
                            items = recognize(cropped_image_path)
                            label = None

                            # Get most likely item that's within the expected classes
                            for item in items:
                                if item["id"] in selected_group:
                                    label = item["id"]
                                    break

                            if label is not None:
                                initial_bricks.append(label)

                        # Skip images with no result   
                        except:
                            continue

                    """
                    Determine correct group
                    """

                    count_1 = 0
                    count_2 = 0
                    count_3 = 0
                    count_4 = 0
                    count_5 = 0
                    count_6 = 0
                    count_7 = 0
                    count_8 = 0
                    count_9 = 0
                    count_10 = 0
                    count_11 = 0
                    count_12 = 0
                    count_13 = 0

                    for id in initial_bricks:
                        if id in group_1:
                            count_1 += 1
                        elif id in group_2:
                            count_2 += 1
                        elif id in group_3:
                            count_3 += 1
                        elif id in group_4:
                            count_4 += 1
                        elif id in group_5:
                            count_5 += 1
                        elif id in group_6:
                            count_6 += 1
                        elif id in group_7:
                            count_7 += 1
                        elif id in group_8:
                            count_8 += 1
                        elif id in group_9:
                            count_9 += 1
                        elif id in group_10:
                            count_10 += 1
                        elif id in group_11:
                            count_11 += 1
                        elif id in group_12:
                            count_12 += 1
                        elif id in group_13:
                            count_13 += 1
                    
                    counts = [count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9, count_10, count_11, count_12, count_13]
                    max_value = max(counts)
                    max_index = counts.index(max_value)

                    selected_group = groups[max_index]   
                    overlap_threshold = overlap_thresholds[max_index]   

                    print("Selected group:", max_index + 1)
                    print("Overlap threshold:", overlap_threshold)

            """
            Redo predictions with correct overlap threshold and group
            """

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
                predictions = remove_overlaps(predictions, overlap_threshold)

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
                predictions = remove_overlaps(predictions, overlap_threshold)

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

                    """
                    Query brickognize
                    """

                    total = 0

                    for cropped_file in cropped_files:
                        try: 
                            cropped_image_path = os.path.abspath(os.path.join(cropped_root, cropped_file))
                            items = recognize(cropped_image_path)
                            label = None
                            name = None
                            img = None

                            # Get most likely item that's within the expected classes
                            for item in items:
                                if item["id"] in selected_group:
                                    label = item["id"]
                                    name = item["name"]
                                    img = item["img_url"]
                                    break

                            if label is not None:
                                # Color detection for similar pieces
                                if label == '43093' or label == '3749':
                                    image = cv2.imread(cropped_image_path)
                                    color = detect_color(image)
                                    if color == 'Yellow':
                                        label = "3749"
                                        name = "Technic, Axle 1L with Pin without Friction Ridges"
                                        img = "https://storage.googleapis.com/brickognize-static/thumbnails-v2.4/part/3749/0.webp"
                                    else:
                                        label = "43093"
                                        name = "Technic, Axle 1L with Pin with Friction Ridges"
                                        img = "https://storage.googleapis.com/brickognize-static/thumbnails-v2.4/part/43093/0.webp"

                                # Add to json
                                if label in bricks:
                                    bricks[label]["count"] += 1
                                else:
                                    bricks[label] = {"count": 1, "name": name, "image_url": img}

                                total += 1

                        # Skip images with no result   
                        except:
                            continue
                    

        # Convert bricks dictionary to JSON format
        json_data = [{"label": label, "name": data["name"], "count": str(data["count"]), "image_url": data["image_url"]} for label, data in bricks.items()]

        # Print final prediction count for debug
        print("Final count:", total)
        print('=' * len(info))

        return jsonify(json_data)

# Default endpoint
@app.route('/')
def home():
    return 'NanoBrick is running! Use the /predict POST endpoint to perform brick predictions.'

# Run app for debug
if __name__ == '__main__':
    # Debug: ./ngrok http --domain=penguin-gorgeous-vaguely.ngrok-free.app 5000
    app.run(debug=True, port=5000)