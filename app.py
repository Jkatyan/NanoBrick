"""
Flask back-end for NanoBrick
CS 4704, Spring 2024
"""

import os
import requests
import tempfile
from PIL import Image
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

"""
Globals
"""

# Class list
classes = ['10928', '2465', '2780', '32009', '32014', '32034', '32054', '32062', '32072', '32073', '32140', '32184', '32248', '32270', '32271', '32291', '32348', '32449', '32498', '32523', '32526', '32556', '33299', '3647', '3648', '3649', '3673', '3706', '3713', '3737', '3749', '40490', '41239', '41678', '4185c01', '42003', '44809', '4519', '45590', '4716', '48989', '55615', '57585', '60483', '60484', '62462', '63869', '64178', '64179', '6536', '6538c', '6558', '6587', '6589', '6629', '81', '82', '83', '84', '85', '87083', '92911', '94925', '99010', '99773', 'x346']

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
    # TODO remove overlaps
    return predictions
    
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

            # Save cropped images
            for prediction in predictions:
                cropped_image = image.crop(prediction['coordinates'])
                cropped_image = cropped_image.convert("RGB")
                cropped_image.save(f"{cropped_output_dir}/{prediction['name']}.jpg")
            
            # Query brickognize
            for cropped_root, _, cropped_files in os.walk(cropped_output_dir):
                for cropped_file in cropped_files:
                    try: 
                        cropped_image_path = os.path.abspath(os.path.join(cropped_root, cropped_file))
                        items = recognize(cropped_image_path)

                        # Take most likely prediction
                        label = items[0]["id"]
                        name = items[0]["name"]
                        img = items[0]["img_url"]

                        # Verify that this (or a similar piece) is in our classes list
                        valid = False
                        for item in items:
                            new_label = item["id"]
                            if new_label in classes:
                                valid = True
                                break
                        
                        if valid:
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