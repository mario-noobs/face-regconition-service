from model import *
from abc import ABC, abstractmethod
from model.request import CreateFaceFeatureRequest
from model.response import CreateFaceFeatureResponse

class FaceFeatureExtractionInterface(ABC):
    
    @abstractmethod
    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
        pass

class FaceHandler(FaceFeatureExtractionInterface):
    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
        response = CreateFaceFeatureResponse()
        response.code = "Success"
        response.message = "Success Message"
        return response
    