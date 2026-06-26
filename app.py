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

def parse_face_request(data, request_cls):
    """Build a face request object from request data, validating required fields."""
    req = request_cls()
    req.user_id = data.get("userId")
    req.request_id = data.get("requestId")
    req.flow = data.get("flow")
    req.image_base64 = data.get("imageBase64")
    req.alg_det = data.get("algorithmDet")
    req.alg_reg = data.get("algorithmReg")

    if not req.user_id or not req.image_base64:
        logger.warning("Missing userId or image in request.")
        raise FaceFeatureException(Messages.MISSING_FIELDS)

    if not req.alg_det or not req.alg_reg:
        logger.warning("Missing algorithm in request.")
        raise FaceFeatureException(Messages.MISSING_ALG)

    logger.info(req.to_dict())
    return req

@app.route('/face/create-identity', methods=['POST'])
def add_identity():
    response = CreateFaceFeatureResponse()
    try:
        req = parse_face_request(request.json, CreateFaceFeatureRequest)
        if not is_valid_base64_image(req.image_base64):
            logger.warning("Invalid Base64 Image.")
            raise FaceFeatureException(Messages.IMAGE_BASE64_ERROR)
        response = face_handler.create_feature(req)
    except FaceFeatureException as ce:
        logger.error("FaceFeatureException processing request: %s", ce)
        response.code = ce.code
        response.message = ce.message
    except Exception as e:
        logger.error("Error processing request: %s", e)
        response.code = Messages.GENERIC_ERROR['code']
        response.message = str(e)
    logger.info(response.to_dict())
    return jsonify(response.to_dict())

@app.route('/face/recognize', methods=['POST'])
def predict():
    response = RecognizeFaceFeatureResponse()
    try:
        req = parse_face_request(request.json, RecognizeFaceFeatureRequest)
        response = face_handler.face_search(req)
    except FaceFeatureException as ce:
        logger.error("FaceFeatureException processing request: %s", ce)
        response.code = ce.code
        response.message = ce.message
    except Exception as e:
        logger.error("Error processing request: %s", e)
        response.code = Messages.GENERIC_ERROR['code']
        response.message = str(e)
    logger.info(response.to_dict())
    return jsonify(response.to_dict())

@app.route('/face/delete-identity', methods=['DELETE'])
def delete_identity():
    """Delete face identity from Redis"""
    try:
        data = request.json
        userId = data.get("userId")
        algorithm = data.get("algorithm", "mobilenet")  # Default to mobilenet
        requestId = data.get("requestId")
        
        # Log request in JSON format
        logger.info(json.dumps({
            "event": "delete_identity_request",
            "requestId": requestId,
            "userId": userId,
            "algorithm": algorithm
        }))
        
        if not userId:
            logger.warning("Missing userId in delete request.")
            return jsonify({
                "status": "error",
                "message": "userId is required",
                "request_id": requestId
            }), 400
        
        logger.info(f"Deleting face identity for userId: {userId}, algorithm: {algorithm}")
        
        # Call the handler method
        deletion_result = face_handler.delete_feature(userId, algorithm)
        
        # Add request_id to the response
        deletion_result["request_id"] = requestId
        
        if deletion_result["status"] == "success":
            logger.info(json.dumps({
                "event": "delete_identity_response",
                "requestId": requestId,
                "status": "success",
                "userId": userId,
                "response": deletion_result
            }))
            return jsonify(deletion_result), 200
        else:
            logger.warning(json.dumps({
                "event": "delete_identity_response",
                "requestId": requestId,
                "status": "failed",
                "userId": userId,
                "response": deletion_result
            }))
            return jsonify(deletion_result), 404
            
    except Exception as e:
        error_response = {
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "request_id": data.get("requestId") if 'data' in locals() else None
        }
        logger.error(json.dumps({
            "event": "delete_identity_error",
            "requestId": data.get("requestId") if 'data' in locals() else None,
            "error": str(e),
            "response": error_response
        }))
        return jsonify(error_response), 500
    
if __name__ == '__main__':
    #app.run(debug=True, host="0.0.0.0")
    app.run(debug=True, host="0.0.0.0", port=5000)
