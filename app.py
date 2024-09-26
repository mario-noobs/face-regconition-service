from tokenize import String
from flask import Flask, jsonify, abort, make_response, request
from pip._vendor.appdirs import unicode
import numpy as np
import base64
import json
from io import BytesIO
import cv2, io
from PIL import Image   
import os, logging
from retinaface import Retinaface
from model.messages import Messages 
from handler.handler import save_image, get_image_paths_and_names


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FACE_DATASET_DIR = "face_dataset"
IMAGE_FORMAT = "JPEG"

@app.route('/face/get-list', methods=['GET'])
def get_names():
    known_face_names = np.load("model_data/{backbone}_names.npy".format(backbone="mobilenet"))
    print(known_face_names)
    return jsonify({'names': list(known_face_names)})

@app.route('/face/add-identity', methods=['POST'])
def add_identity(): 
    try:

        data = request.json
        name = (data.get("name"))
        image = (data.get("image"))
        
        if not name or not image:
            logger.warning("Missing name or image in request.")
            return jsonify({'code': Messages.MISSING_FIELDS['code'], 
                            'message': Messages.MISSING_FIELDS['message']}), 400

        filename = os.path.join(FACE_DATASET_DIR, f"{name}_1.jpg")
        save_image(image, filename)

        retinaface = Retinaface(1)

        image_paths, names = get_image_paths_and_names(FACE_DATASET_DIR)

        retinaface.encode_face_dataset(image_paths,names)

        code = Messages.SUCCESS['code']

        response_message = Messages.SUCCESS['message']

    except:
        logger.error("Error processing request: %s", e)
        code = Messages.GENERIC_ERROR['code']
        response_message = Messages.GENERIC_ERROR['message']

    logger.info("Response code: %s", code)

    return jsonify({'code': code, 'message': response_message})

@app.route('/face/recognize', methods=['POST']) 
def predict():
    try:
        retinaface = Retinaface()
        data = request.json
        response = {}
        image = (data.get("image"))
        if (image!=None):
            imgdata = base64.b64decode(image)
            img = Image.open(io.BytesIO(imgdata))
            image  = cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)

        r_image, data = retinaface.detect_image(image)

        cv2.imwrite("face.jpg", r_image)
        with open("face.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
       
        print(data)
        response["code"] = "0000"
        response['confidence'] = data
        response['raw_image'] = encoded_string
    except:
        response["code"] = "1111"
        response['confidence'] = "NONE"
        response['raw_image'] = "image error"

    return response
    
if __name__ == '__main__':
    #app.run(debug=True, host="0.0.0.0")
    app.run(host="0.0.0.0", port=5000)
