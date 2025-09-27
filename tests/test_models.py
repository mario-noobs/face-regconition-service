import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.request import CreateFaceFeatureRequest, RecognizeFaceFeatureRequest
from model.response import CreateFaceFeatureResponse, RecognizeFaceFeatureResponse, BaseResponse
from model.exceptions import FaceFeatureException
from model.messages import Messages


class TestModels(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass

    def test_create_face_feature_request_initialization(self):
        """Test CreateFaceFeatureRequest initialization"""
        request = CreateFaceFeatureRequest()
        
        self.assertIsNone(request.user_id)
        self.assertIsNone(request.image_base64)
        self.assertIsNone(request.flow)
        self.assertIsNone(request.name)
        self.assertIsNone(request.request_id)
        self.assertIsNone(request.alg_det)
        self.assertIsNone(request.alg_reg)

    def test_create_face_feature_request_to_dict(self):
        """Test CreateFaceFeatureRequest to_dict method"""
        request = CreateFaceFeatureRequest()
        request.user_id = "test_user_123"
        request.image_base64 = "sample_base64_image"
        request.flow = "registration"
        request.name = "John Doe"
        request.request_id = "req_123"
        request.alg_det = "retinaface"
        request.alg_reg = "mobilenet"
        
        result = request.to_dict()
        
        self.assertEqual(result["user_id"], "test_user_123")
        self.assertEqual(result["flow"], "registration")
        self.assertEqual(result["name"], "John Doe")
        self.assertEqual(result["request_id"], "req_123")
        self.assertEqual(result["alg_det"], "retinaface")
        self.assertEqual(result["alg_reg"], "mobilenet")
        # image_base64_hash should be present and not equal to original
        self.assertIsNotNone(result["image_base64_hash"])
        self.assertNotEqual(result["image_base64_hash"], "sample_base64_image")

    def test_recognize_face_feature_request_initialization(self):
        """Test RecognizeFaceFeatureRequest initialization"""
        request = RecognizeFaceFeatureRequest()
        
        self.assertIsNone(request.user_id)
        self.assertIsNone(request.image_base64)
        self.assertIsNone(request.flow)
        self.assertIsNone(request.name)
        self.assertIsNone(request.request_id)
        self.assertIsNone(request.alg_det)
        self.assertIsNone(request.alg_reg)

    def test_recognize_face_feature_request_to_dict(self):
        """Test RecognizeFaceFeatureRequest to_dict method"""
        request = RecognizeFaceFeatureRequest()
        request.user_id = "test_user_456"
        request.image_base64 = "another_base64_image"
        request.flow = "recognition"
        request.name = "Jane Doe"
        request.request_id = "req_456"
        request.alg_det = "retinaface"
        request.alg_reg = "mobilenet"
        
        result = request.to_dict()
        
        self.assertEqual(result["user_id"], "test_user_456")
        self.assertEqual(result["flow"], "recognition")
        self.assertEqual(result["name"], "Jane Doe")
        self.assertEqual(result["request_id"], "req_456")
        self.assertEqual(result["alg_det"], "retinaface")
        self.assertEqual(result["alg_reg"], "mobilenet")
        # image_base64_hash should be present and not equal to original
        self.assertIsNotNone(result["image_base64_hash"])
        self.assertNotEqual(result["image_base64_hash"], "another_base64_image")

    def test_base_response_initialization(self):
        """Test BaseResponse initialization"""
        response = BaseResponse()
        
        self.assertIsNone(response.code)
        self.assertIsNone(response.message)

    def test_base_response_to_dict(self):
        """Test BaseResponse to_dict method"""
        response = BaseResponse()
        response.code = "0000"
        response.message = "Success"
        
        result = response.to_dict()
        
        self.assertEqual(result["code"], "0000")
        self.assertEqual(result["message"], "Success")

    def test_create_face_feature_response_initialization(self):
        """Test CreateFaceFeatureResponse initialization"""
        response = CreateFaceFeatureResponse()
        
        self.assertIsNone(response.code)
        self.assertIsNone(response.message)
        self.assertIsNone(response.user_id)
        self.assertIsNone(response.request_id)
        self.assertIsNone(response.face_encoding_base64)
        self.assertIsNone(response.encoding_shape)

    def test_create_face_feature_response_set_data(self):
        """Test CreateFaceFeatureResponse set_data method"""
        response = CreateFaceFeatureResponse()
        response.set_data(123, "req_123", "sample_feature")
        
        self.assertEqual(response.user_id, 123)
        self.assertEqual(response.request_id, "req_123")
        self.assertEqual(response.feature, "sample_feature")

    def test_create_face_feature_response_to_dict(self):
        """Test CreateFaceFeatureResponse to_dict method"""
        response = CreateFaceFeatureResponse()
        response.code = "0000"
        response.message = "Success"
        response.user_id = "test_user_123"
        response.request_id = "req_123"
        response.face_encoding_base64 = "encoded_features"
        response.encoding_shape = [1, 512]
        
        result = response.to_dict()
        
        self.assertEqual(result["code"], "0000")
        self.assertEqual(result["message"], "Success")
        self.assertEqual(result["data"]["user_id"], "test_user_123")
        self.assertEqual(result["data"]["request_id"], "req_123")
        self.assertEqual(result["data"]["face_encoding_base64"], "encoded_features")
        self.assertEqual(result["data"]["encoding_shape"], [1, 512])

    def test_recognize_face_feature_response_initialization(self):
        """Test RecognizeFaceFeatureResponse initialization"""
        response = RecognizeFaceFeatureResponse()
        
        self.assertIsNone(response.code)
        self.assertIsNone(response.message)
        self.assertIsNone(response.user_id)
        self.assertIsNone(response.request_id)
        self.assertIsNone(response.searh_result)  # Note: keeping the typo as in original

    def test_recognize_face_feature_response_to_dict(self):
        """Test RecognizeFaceFeatureResponse to_dict method"""
        response = RecognizeFaceFeatureResponse()
        response.code = "0000"
        response.message = "Success"
        response.user_id = "test_user_456"
        response.request_id = "req_456"
        response.searh_result = {"match_score": 0.95, "matched_user": "test_user_123"}
        
        result = response.to_dict()
        
        self.assertEqual(result["code"], "0000")
        self.assertEqual(result["message"], "Success")
        self.assertEqual(result["data"]["user_id"], "test_user_456")
        self.assertEqual(result["data"]["request_id"], "req_456")
        self.assertEqual(result["data"]["searh_result"]["match_score"], 0.95)
        self.assertEqual(result["data"]["searh_result"]["matched_user"], "test_user_123")

    def test_face_feature_exception_with_messages(self):
        """Test FaceFeatureException with Messages"""
        exception = FaceFeatureException(Messages.NO_FACE)
        
        self.assertEqual(exception.code, Messages.NO_FACE['code'])
        self.assertEqual(exception.message, Messages.NO_FACE['message'])

    def test_face_feature_exception_custom_message(self):
        """Test FaceFeatureException with custom message"""
        custom_message = {"code": "9999", "message": "Custom error"}
        exception = FaceFeatureException(custom_message)
        
        self.assertEqual(exception.code, "9999")
        self.assertEqual(exception.message, "Custom error")

    def test_messages_constants(self):
        """Test Messages class constants"""
        self.assertEqual(Messages.SUCCESS['code'], "0000")
        self.assertEqual(Messages.MISSING_FIELDS['code'], "4000")
        self.assertEqual(Messages.MISSING_ALG['code'], "4002")
        self.assertEqual(Messages.IMAGE_BASE64_ERROR['code'], "4001")
        self.assertEqual(Messages.GENERIC_ERROR['code'], "1111")
        self.assertEqual(Messages.IMAGE_SAVE_ERROR['code'], "5000")
        self.assertEqual(Messages.IMAGE_ENCODING_ERROR['code'], "5001")
        self.assertEqual(Messages.NO_FACE['code'], "5002")
        
        # Verify messages are strings
        self.assertIsInstance(Messages.SUCCESS['message'], str)
        self.assertIsInstance(Messages.MISSING_FIELDS['message'], str)
        self.assertIsInstance(Messages.NO_FACE['message'], str)


if __name__ == '__main__':
    unittest.main()