import json
import logging

from flask import Flask, jsonify, request

from handler.handlers import FaceHandler
from model.exceptions import FaceFeatureException
from model.messages import Messages
from model.request import CreateFaceFeatureRequest, RecognizeFaceFeatureRequest
from model.response import CreateFaceFeatureResponse, RecognizeFaceFeatureResponse
from utils.helpers import is_valid_base64_image

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_handler = FaceHandler(logger=logger)


def _parse_face_request(data):
    """Extract and validate common fields from a face API request body.

    Raises FaceFeatureException for missing required fields.
    Returns (user_id, image, flow, request_id, algorithm_det, algorithm_reg).
    """
    user_id       = data.get("userId")
    image         = data.get("imageBase64")
    flow          = data.get("flow")
    request_id    = data.get("requestId")
    algorithm_det = data.get("algorithmDet")
    algorithm_reg = data.get("algorithmReg")

    if not user_id or not image:
        logger.warning("Missing userId or image in request.")
        raise FaceFeatureException(Messages.MISSING_FIELDS)

    if not algorithm_det or not algorithm_reg:
        logger.warning("Missing algorithm in request.")
        raise FaceFeatureException(Messages.MISSING_ALG)

    return user_id, image, flow, request_id, algorithm_det, algorithm_reg


@app.route('/face/create-identity', methods=['POST'])
def add_identity():
    response = CreateFaceFeatureResponse()
    try:
        user_id, image, flow, request_id, algorithm_det, algorithm_reg = _parse_face_request(request.json)

        req = CreateFaceFeatureRequest()
        req.user_id       = user_id
        req.request_id    = request_id
        req.flow          = flow
        req.image_base64  = image
        req.alg_det       = algorithm_det
        req.alg_reg       = algorithm_reg

        logger.info(req.to_dict())

        if not is_valid_base64_image(image):
            logger.warning("Invalid Base64 Image.")
            raise FaceFeatureException(Messages.IMAGE_BASE64_ERROR)

        response = face_handler.create_feature(req)

    except FaceFeatureException as ce:
        logger.error("FaceFeatureException processing request: %s", ce)
        response.code    = ce.code
        response.message = ce.message

    except Exception as e:
        logger.error("Error processing request: %s", e)
        response.code    = Messages.GENERIC_ERROR['code']
        response.message = str(e)

    logger.info(response.to_dict())
    return jsonify(response.to_dict())


@app.route('/face/recognize', methods=['POST'])
def predict():
    response = RecognizeFaceFeatureResponse()
    try:
        user_id, image, flow, request_id, algorithm_det, algorithm_reg = _parse_face_request(request.json)

        req = RecognizeFaceFeatureRequest()
        req.user_id       = user_id
        req.request_id    = request_id
        req.flow          = flow
        req.image_base64  = image
        req.alg_det       = algorithm_det
        req.alg_reg       = algorithm_reg

        logger.info(req.to_dict())

        response = face_handler.face_search(req)

    except FaceFeatureException as ce:
        logger.error("FaceFeatureException processing request: %s", ce)
        response.code    = ce.code
        response.message = ce.message

    except Exception as e:
        logger.error("Error processing request: %s", e)
        response.code    = Messages.GENERIC_ERROR['code']
        response.message = str(e)

    logger.info(response.to_dict())
    return jsonify(response.to_dict())


@app.route('/face/delete-identity', methods=['DELETE'])
def delete_identity():
    """Delete face identity from Redis."""
    try:
        data       = request.json
        user_id    = data.get("userId")
        algorithm  = data.get("algorithm", "mobilenet")
        request_id = data.get("requestId")

        logger.info(json.dumps({
            "event": "delete_identity_request",
            "requestId": request_id,
            "userId": user_id,
            "algorithm": algorithm
        }))

        if not user_id:
            logger.warning("Missing userId in delete request.")
            return jsonify({
                "status": "error",
                "message": "userId is required",
                "request_id": request_id
            }), 400

        deletion_result = face_handler.delete_feature(user_id, algorithm)
        deletion_result["request_id"] = request_id

        status_code = 200 if deletion_result["status"] == "success" else 404
        logger.info(json.dumps({
            "event": "delete_identity_response",
            "requestId": request_id,
            "status": deletion_result["status"],
            "userId": user_id,
            "response": deletion_result
        }))
        return jsonify(deletion_result), status_code

    except Exception as e:
        request_id = data.get("requestId") if 'data' in locals() else None
        error_response = {
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "request_id": request_id
        }
        logger.error(json.dumps({
            "event": "delete_identity_error",
            "requestId": request_id,
            "error": str(e),
            "response": error_response
        }))
        return jsonify(error_response), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
