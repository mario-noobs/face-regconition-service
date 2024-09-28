from tokenize import String
from flask import Flask, jsonify, abort, make_response, request
import numpy as np
import base64
import json
from io import BytesIO
import cv2, io
from PIL import Image   
import os, logging
from retinaface import Retinaface
from model.messages import Messages
from model import *
from utils.helpers import is_valid_base64_image, convert_string_to_hash
import hashlib
import logging
from PIL import Image
from io import BytesIO
from handler.handlers import FaceHandler, FaceFeatureExtractionInterface
from model.request import CreateFaceFeatureRequest, RecognizeFaceFeatureRequest
from model.response import CreateFaceFeatureResponse, RecognizeFaceFeatureResponse
from model.exceptions import FaceFeatureException

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FACE_DATASET_DIR = "face_dataset"
IMAGE_FORMAT = "JPEG"

face_handler = FaceHandler(logger = logger)

@app.route('/face/create-identity', methods=['POST'])
def add_identity(): 
    try:

        data = request.json
        
        userId = (data.get("userId"))
        image = (data.get("imageBase64"))
        flow = (data.get("flow"))
        requestId = (data.get("requestId"))
        algorithm_det = (data.get("algorithmDet"))
        algorithm_reg = (data.get("algorithmReg"))

        createFaceFeatureResponse = CreateFaceFeatureResponse()
        
        if not userId or not image:
            logger.warning("Missing userId or image in request.")
            raise FaceFeatureException(Messages.MISSING_FIELDS)

        if not algorithm_det or not algorithm_reg:
            logger.warning("Missing algorithm in request.")
            raise FaceFeatureException(Messages.MISSING_ALG)  

        createFaceFeatureRequest = CreateFaceFeatureRequest()
        createFaceFeatureRequest.user_id = userId
        createFaceFeatureRequest.request_id = requestId
        createFaceFeatureRequest.flow = flow
        createFaceFeatureRequest.image_base64 = image
        createFaceFeatureRequest.alg_det = algorithm_det
        createFaceFeatureRequest.alg_reg = algorithm_reg        
      
        logger.info(createFaceFeatureRequest.to_dict())

        if not is_valid_base64_image(image):
            logger.warning("Invalid Base64 Image.")
            raise FaceFeatureException(Messages.IMAGE_BASE64_ERROR) 

        createFaceFeatureResponse = face_handler.create_feature(createFaceFeatureRequest)

    except FaceFeatureException as ce:
        logger.error("FaceFeatureException processing request: %s", ce)
        createFaceFeatureResponse.code = ce.code
        createFaceFeatureResponse.message = ce.message

    except Exception as e:
        logger.error("Error processing request: %s", e)
        createFaceFeatureResponse.code = Messages.GENERIC_ERROR['code']
        createFaceFeatureResponse.message = str(e)

    logger.info(createFaceFeatureResponse.to_dict())

    return jsonify(createFaceFeatureResponse.to_dict())

@app.route('/face/recognize', methods=['POST'])
def predict():
    try:
        
        data = request.json
        userId = (data.get("userId"))
        image = (data.get("imageBase64"))
        flow = (data.get("flow"))
        requestId = (data.get("requestId"))
        algorithm_det = (data.get("algorithmDet"))
        algorithm_reg = (data.get("algorithmReg"))

        recognizeFaceFeatureResponse = RecognizeFaceFeatureResponse()
        
        if not userId or not image:
            logger.warning("Missing userId or image in request.")
            raise FaceFeatureException(Messages.MISSING_FIELDS)

        if not algorithm_det or not algorithm_reg:
            logger.warning("Missing algorithm in request.")
            raise FaceFeatureException(Messages.MISSING_ALG)  

        recognizeFaceFeatureRequest = RecognizeFaceFeatureRequest()
        recognizeFaceFeatureRequest.user_id = userId
        recognizeFaceFeatureRequest.request_id = requestId
        recognizeFaceFeatureRequest.flow = flow
        recognizeFaceFeatureRequest.image_base64 = image
        recognizeFaceFeatureRequest.alg_det = algorithm_det
        recognizeFaceFeatureRequest.alg_reg = algorithm_reg        
      
        logger.info(recognizeFaceFeatureRequest.to_dict())

        recognizeFaceFeatureResponse = face_handler.face_search(recognizeFaceFeatureRequest)

    except FaceFeatureException as ce:
        logger.error("FaceFeatureException processing request: %s", ce)
        recognizeFaceFeatureResponse.code = ce.code
        recognizeFaceFeatureResponse.message = ce.message

    except Exception as e:
        logger.error("Error processing request: %s", e)
        recognizeFaceFeatureResponse.code = Messages.GENERIC_ERROR['code']
        recognizeFaceFeatureResponse.message = str(e)

    logger.info(recognizeFaceFeatureResponse.to_dict())

    return jsonify(recognizeFaceFeatureResponse.to_dict())
    
if __name__ == '__main__':
    #app.run(debug=True, host="0.0.0.0")
    app.run(debug=True, host="0.0.0.0", port=5000)
