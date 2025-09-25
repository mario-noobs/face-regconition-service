class BaseResponse:
    def __init__(self):
        self.code = None
        self.message = None

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
        }

class CreateFaceFeatureResponse(BaseResponse):
    def __init__(self):
        super().__init__()
        self.user_id = None
        self.request_id = None
        self.face_encoding_base64 = None  # Add this field
        self.encoding_shape = None        # Add this field

    def set_data(self, user_id: int, request_id: str, feature):
        self.user_id = user_id
        self.request_id = request_id
        self.feature = feature

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "data": {
                "user_id": self.user_id,
                "request_id": self.request_id,
                "face_encoding_base64": self.face_encoding_base64,
                "encoding_shape": self.encoding_shape
            }
        })
        return base_dict

class RecognizeFaceFeatureResponse(BaseResponse):
    def __init__(self):
        super().__init__()
        self.user_id = None
        self.request_id = None
        self.searh_result = None

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "data": {
                "user_id": self.user_id,
                "request_id": self.request_id,
                "searh_result": self.searh_result,
            }
        })
        return base_dict