from model import *
from abc import ABC, abstractmethod
from model.request import CreateFaceFeatureRequest, RecognizeFaceFeatureRequest
from model.response import CreateFaceFeatureResponse, RecognizeFaceFeatureResponse
from model.exceptions import FaceFeatureException
from retinaface import Retinaface
from model.messages import Messages

class FaceFeatureExtractionInterface(ABC):
    
    @abstractmethod
    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
        pass
    
    @abstractmethod
    def face_search(self, request: RecognizeFaceFeatureRequest) -> RecognizeFaceFeatureResponse:
        pass
    
    @abstractmethod
    def delete_feature(self, user_id: str, algorithm: str = None) -> dict:
        pass

class FaceHandler(FaceFeatureExtractionInterface):
    def __init__(self, config = None, logger = None):
        self.retinaface = Retinaface()
        self.config = config
        self.logger = logger

    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
        """Create face feature encoding and store in Redis"""
        response = CreateFaceFeatureResponse()
        response.request_id = request.request_id
        response.user_id = request.user_id

        try:
            self.logger.info(f"Creating face feature for user_id: {request.user_id}, algorithm: {request.alg_reg}")
            
            encoding_result = self.retinaface.encode_face_image(request.user_id, request.image_base64, request.alg_reg)

            if encoding_result["status"] == 0:
                self.logger.warning(f"No face detected for user_id: {request.user_id}")
                raise FaceFeatureException(Messages.NO_FACE)

            if encoding_result["status"] < 0:
                self.logger.error(f"Encoding error for user_id: {request.user_id}, error: {encoding_result.get('message')}")
                raise FaceFeatureException(Messages.IMAGE_ENCODING_ERROR)

            response.face_encoding_base64 = encoding_result.get("face_encoding_base64")
            response.encoding_shape = encoding_result.get("encoding_shape")
            self.logger.info(f"Face feature created successfully for user_id: {request.user_id}")

            response.code = Messages.SUCCESS["code"]
            response.message = Messages.SUCCESS["message"]

        except FaceFeatureException as fe:
            self.logger.error(f"FaceFeatureException in create_feature for user_id {request.user_id}: {fe}")
            response.code = fe.code
            response.message = fe.message

        except Exception as e:
            self.logger.error(f"Exception in create_feature for user_id {request.user_id}: {e}")
            response.code = Messages.GENERIC_ERROR["code"]
            response.message = str(e)
            
        return response
    
    def face_search(self, request: RecognizeFaceFeatureRequest) -> RecognizeFaceFeatureResponse:
        """Search for face in the stored encodings"""
        response = RecognizeFaceFeatureResponse()
        response.request_id = request.request_id
        response.user_id = request.user_id

        try:
            self.logger.info(f"Searching face for user_id: {request.user_id}, algorithm: {request.alg_reg}")
            
            # Reload face features from Redis
            self.retinaface.reload_face_feature(request.alg_reg)

            # Search for face in the image
            _, data = self.retinaface.search_face(request.image_base64)
        
            if not data:
                self.logger.warning(f"No face found in search for user_id: {request.user_id}")
                raise FaceFeatureException(Messages.NO_FACE)

            response.code = Messages.SUCCESS["code"]
            response.message = Messages.SUCCESS["message"]
            response.searh_result = data
            
            self.logger.info(f"Face search completed successfully for user_id: {request.user_id}")

        except FaceFeatureException as fe:
            self.logger.error(f"FaceFeatureException in face_search for user_id {request.user_id}: {fe}")
            response.code = fe.code
            response.message = fe.message

        except Exception as e:
            self.logger.error(f"Exception in face_search for user_id {request.user_id}: {e}")
            response.code = Messages.GENERIC_ERROR["code"]
            response.message = str(e)
            
        return response

    def delete_feature(self, user_id: str, algorithm: str = None) -> dict:
        """Delete face feature encoding from Redis by userId"""
        result = {
            "status": "success",
            "message": "Face feature deleted successfully",
            "user_id": user_id
        }
        
        try:
            self.logger.info(f"Attempting to delete face feature for user_id: {user_id}, algorithm: {algorithm}")
            
            # Call the retinaface method to delete from Redis
            deleted = self.retinaface.delete_face_data(user_id, algorithm)
            if not deleted:
                result["status"] = "error"
                result["message"] = f"Face feature not found for user_id: {user_id}"
                self.logger.warning(f"Face feature not found for deletion: {user_id}")
            self.logger.info(f"Face feature deletion completed for {user_id}: {result['status']}")
            
        except Exception as e:
            self.logger.error(f"Exception deleting face feature for {user_id}: {e}")
            result["status"] = "error"
            result["message"] = f"Error deleting face feature: {str(e)}"
        
        return result
