class CreateFaceFeatureRequest():
    def __init__(self):
        self.user_id = None
        self.image_base64 = None
        self.flow = None
        self.request_id = None
        self.alg_det = None
        self.alg_reg = None

    def to_dict(self):
        base_dict.update({
            "user_id": self.user_id,
            "image_base64": self.image_base64,
            "flow": self.flow,
            "request_id": self.request_id,
            "alg_det": self.alg_det,
            "alg_reg": self.alg_reg,
        })
        return base_dict

