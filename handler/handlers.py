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

class FaceHandler(FaceFeatureExtractionInterface):
    def __init__(self, config = None, logger = None):
        self.retinaface = Retinaface()
        self.config = config
        self.logger = logger

    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
        response = CreateFaceFeatureResponse()
        response.request_id = request.request_id
        response.user_id = request.user_id

        try:
            create_success = self.retinaface.encode_face_image(request.user_id, request.image_base64, request.alg_reg)
            
            if create_success == 0:
                raise FaceFeatureException(Messages.NO_FACE)

            if create_success < 0:
                raise FaceFeatureException(Messages.IMAGE_ENCODING_ERROR)

            response.code = Messages.SUCCESS["code"]
            response.message = Messages.SUCCESS["message"]

        except FaceFeatureException as fe:
            self.logger.error("FaceFeatureException create feature error: %s", fe)
            response.code = fe.code
            response.message = fe.message

        except Exception as e:
            raise Exception(e)
        return response
    
    def face_search(self, request: RecognizeFaceFeatureRequest) -> RecognizeFaceFeatureResponse:
        response = RecognizeFaceFeatureResponse()
        response.request_id = request.request_id
        response.user_id = request.user_id

        try:
            self.retinaface.reload_face_feature(request.alg_reg)

            _ , data = self.retinaface.search_face(request.image_base64)
        
            if not data:
                raise FaceFeatureException(Messages.NO_FACE)

            response.code = Messages.SUCCESS["code"]
            response.message = Messages.SUCCESS["message"]
            response.searh_result = data

        except FaceFeatureException as fe:
            self.logger.error("FaceFeatureException searching face error: %s", fe)
            response.code = fe.code
            response.message = fe.message

        except Exception as e:
            raise Exception(e)
            
        return response
    