from model import *
from abc import ABC, abstractmethod
from model.request import CreateFaceFeatureRequest
from model.response import CreateFaceFeatureResponse
from model.exceptions import FaceFeatureException
from retinaface import Retinaface
from model.messages import Messages

class FaceFeatureExtractionInterface(ABC):
    
    @abstractmethod
    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
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
    