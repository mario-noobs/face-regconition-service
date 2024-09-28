from model.messages import *

class FaceFeatureException(Exception):
    def __init__(self, message: dict):
        self.code = message['code']
        self.message = message['message']
        super().__init__(self.message)