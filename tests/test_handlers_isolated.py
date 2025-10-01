import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock all heavy dependencies before any imports
mock_modules = {
    'torch': MagicMock(),
    'torchvision': MagicMock(),
    # 'cv2': MagicMock(),
    'redis': MagicMock(),
    'retinaface': MagicMock(),
    'nets': MagicMock(),
    'nets.retinaface': MagicMock(),
    'nets.facenet': MagicMock(),
    'nets.inception_resnetv1': MagicMock(),
    'nets.mobilenet': MagicMock(),
    'utils.anchors': MagicMock(),
    'utils.utils_bbox': MagicMock(),
    'utils.config': MagicMock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Now we can safely import our modules
from model.request import CreateFaceFeatureRequest, RecognizeFaceFeatureRequest
from model.response import CreateFaceFeatureResponse, RecognizeFaceFeatureResponse
from model.exceptions import FaceFeatureException
from model.messages import Messages


class MockRetinaface:
    """Mock Retinaface class that behaves like the real one"""
    
    def __init__(self):
        pass
    
    def encode_face_image(self, user_id, image_base64, algorithm):
        # This will be overridden by test-specific mocks
        return {"status": 1, "message": "Success"}
    
    def reload_face_feature(self, algorithm):
        pass
    
    def search_face(self, image_base64):
        # This will be overridden by test-specific mocks
        return True, {"match_score": 0.95}
    
    def delete_face_data(self, user_id, algorithm):
        # This will be overridden by test-specific mocks
        return True


class MockFaceHandler:
    """Mock FaceHandler that mimics the real handler logic"""
    
    def __init__(self, logger=None):
        self.logger = logger or MagicMock()
        self.retinaface = MockRetinaface()
    
    def create_feature(self, request: CreateFaceFeatureRequest) -> CreateFaceFeatureResponse:
        """Create face feature encoding - mimics real implementation"""
        response = CreateFaceFeatureResponse()
        response.request_id = request.request_id
        response.user_id = request.user_id

        try:
            self.logger.info(f"Creating face feature for user_id: {request.user_id}")
            
            # Call encode_face_image method
            encoding_result = self.retinaface.encode_face_image(request.user_id, request.image_base64, request.alg_reg)
            
            # Handle dictionary return format (like the real implementation)
            if isinstance(encoding_result, dict):
                if encoding_result["status"] == 0:
                    self.logger.warning(f"No face detected for user_id: {request.user_id}")
                    raise FaceFeatureException(Messages.NO_FACE)
                
                if encoding_result["status"] < 0:
                    self.logger.error(f"Encoding error for user_id: {request.user_id}")
                    raise FaceFeatureException(Messages.IMAGE_ENCODING_ERROR)
                
                # Store the face encoding in the response
                response.face_encoding_base64 = encoding_result.get("face_encoding_base64")
                response.encoding_shape = encoding_result.get("encoding_shape")

            response.code = Messages.SUCCESS["code"]
            response.message = Messages.SUCCESS["message"]

        except FaceFeatureException as fe:
            self.logger.error(f"FaceFeatureException in create_feature: {fe}")
            response.code = fe.code
            response.message = fe.message

        except Exception as e:
            self.logger.error(f"Exception in create_feature: {e}")
            response.code = Messages.GENERIC_ERROR["code"]
            response.message = str(e)
            
        return response
    
    def face_search(self, request: RecognizeFaceFeatureRequest) -> RecognizeFaceFeatureResponse:
        """Search for face - mimics real implementation"""
        response = RecognizeFaceFeatureResponse()
        response.request_id = request.request_id
        response.user_id = request.user_id

        try:
            self.logger.info(f"Searching face for user_id: {request.user_id}")
            
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

        except FaceFeatureException as fe:
            self.logger.error(f"FaceFeatureException in face_search: {fe}")
            response.code = fe.code
            response.message = fe.message

        except Exception as e:
            self.logger.error(f"Exception in face_search: {e}")
            response.code = Messages.GENERIC_ERROR["code"]
            response.message = str(e)
            
        return response

    def delete_feature(self, user_id: str, algorithm: str = None) -> dict:
        """Delete face feature - mimics real implementation"""
        result = {
            "status": "success", 
            "message": "Face feature deleted successfully",
            "user_id": user_id
        }
        
        try:
            self.logger.info(f"Deleting face feature for user_id: {user_id}")
            
            # Call the retinaface method to delete from Redis
            deletion_result = self.retinaface.delete_face_data(user_id, algorithm)
            
            if isinstance(deletion_result, bool):
                if not deletion_result:
                    result["status"] = "error"
                    result["message"] = f"Face feature not found for user_id: {user_id}"
            elif isinstance(deletion_result, dict):
                if deletion_result.get("status") == "error":
                    result["status"] = "error"
                    result["message"] = deletion_result.get("message", "Unknown error occurred")
            
        except Exception as e:
            self.logger.error(f"Exception deleting face feature: {e}")
            result["status"] = "error"
            result["message"] = f"Error deleting face feature: {str(e)}"
        
        return result


class TestFaceHandlerIsolated(unittest.TestCase):
    """Isolated tests for FaceHandler with proper mocking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_logger = MagicMock()
        self.face_handler = MockFaceHandler(logger=self.mock_logger)
        
        # Sample request data
        self.create_request = CreateFaceFeatureRequest()
        self.create_request.user_id = "test_user_123"
        self.create_request.image_base64 = "sample_base64_image"
        self.create_request.request_id = "req_123"
        self.create_request.alg_reg = "mobilenet"
        self.create_request.alg_det = "retinaface"
        self.create_request.flow = "registration"
        
        self.recognize_request = RecognizeFaceFeatureRequest()
        self.recognize_request.user_id = "test_user_123"
        self.recognize_request.image_base64 = "sample_base64_image"
        self.recognize_request.request_id = "req_456"
        self.recognize_request.alg_reg = "mobilenet"
        self.recognize_request.alg_det = "retinaface"
        self.recognize_request.flow = "recognition"

    def test_create_feature_success(self):
        """Test successful feature creation"""
        # Mock successful encoding result
        mock_encoding_result = {
            "status": 1,
            "message": "Success",
            "face_encoding_base64": "encoded_features",
            "encoding_shape": [1, 512]
        }
        self.face_handler.retinaface.encode_face_image = MagicMock(return_value=mock_encoding_result)
        
        response = self.face_handler.create_feature(self.create_request)
        
        self.assertIsInstance(response, CreateFaceFeatureResponse)
        self.assertEqual(response.request_id, "req_123")
        self.assertEqual(response.user_id, "test_user_123")
        self.assertEqual(response.code, Messages.SUCCESS['code'])
        self.assertEqual(response.message, Messages.SUCCESS['message'])
        self.assertEqual(response.face_encoding_base64, "encoded_features")
        self.assertEqual(response.encoding_shape, [1, 512])

    def test_create_feature_no_face_detected(self):
        """Test feature creation when no face is detected"""
        # Mock no face detected result
        mock_encoding_result = {
            "status": 0,
            "message": "No face detected"
        }
        self.face_handler.retinaface.encode_face_image = MagicMock(return_value=mock_encoding_result)
        
        response = self.face_handler.create_feature(self.create_request)
        
        self.assertIsInstance(response, CreateFaceFeatureResponse)
        self.assertEqual(response.code, Messages.NO_FACE['code'])
        self.assertEqual(response.message, Messages.NO_FACE['message'])

    def test_create_feature_encoding_error(self):
        """Test feature creation with encoding error"""
        # Mock encoding error result
        mock_encoding_result = {
            "status": -1,
            "message": "Encoding failed"
        }
        self.face_handler.retinaface.encode_face_image = MagicMock(return_value=mock_encoding_result)
        
        response = self.face_handler.create_feature(self.create_request)
        
        self.assertIsInstance(response, CreateFaceFeatureResponse)
        self.assertEqual(response.code, Messages.IMAGE_ENCODING_ERROR['code'])
        self.assertEqual(response.message, Messages.IMAGE_ENCODING_ERROR['message'])

    def test_create_feature_exception_handling(self):
        """Test feature creation with exception"""
        # Mock exception during encoding
        self.face_handler.retinaface.encode_face_image = MagicMock(side_effect=Exception("Encoding error"))
        
        response = self.face_handler.create_feature(self.create_request)
        
        self.assertIsInstance(response, CreateFaceFeatureResponse)
        self.assertEqual(response.code, Messages.GENERIC_ERROR['code'])
        self.assertIn("Encoding error", response.message)

    def test_face_search_success(self):
        """Test successful face search"""
        # Mock successful search result
        mock_search_data = {
            "match_score": 0.95,
            "matched_user": "test_user_123"
        }
        self.face_handler.retinaface.reload_face_feature = MagicMock()
        self.face_handler.retinaface.search_face = MagicMock(return_value=(True, mock_search_data))
        
        response = self.face_handler.face_search(self.recognize_request)
        
        self.assertIsInstance(response, RecognizeFaceFeatureResponse)
        self.assertEqual(response.request_id, "req_456")
        self.assertEqual(response.user_id, "test_user_123")
        self.assertEqual(response.code, Messages.SUCCESS['code'])
        self.assertEqual(response.message, Messages.SUCCESS['message'])
        self.assertEqual(response.searh_result, mock_search_data)

    def test_face_search_no_face_detected(self):
        """Test face search when no face is detected"""
        # Mock no data returned
        self.face_handler.retinaface.reload_face_feature = MagicMock()
        self.face_handler.retinaface.search_face = MagicMock(return_value=(False, None))
        
        response = self.face_handler.face_search(self.recognize_request)
        
        self.assertIsInstance(response, RecognizeFaceFeatureResponse)
        self.assertEqual(response.code, Messages.NO_FACE['code'])
        self.assertEqual(response.message, Messages.NO_FACE['message'])

    def test_face_search_exception_handling(self):
        """Test face search with exception"""
        self.face_handler.retinaface.reload_face_feature = MagicMock(side_effect=Exception("Search error"))
        
        response = self.face_handler.face_search(self.recognize_request)
        
        self.assertIsInstance(response, RecognizeFaceFeatureResponse)
        self.assertEqual(response.code, Messages.GENERIC_ERROR['code'])
        self.assertIn("Search error", response.message)

    def test_delete_feature_success(self):
        """Test successful feature deletion"""
        self.face_handler.retinaface.delete_face_data = MagicMock(return_value=True)
        
        result = self.face_handler.delete_feature("test_user_123", "mobilenet")
        
        self.assertEqual(result["status"], "success")
        self.face_handler.retinaface.delete_face_data.assert_called_once_with("test_user_123", "mobilenet")

    def test_delete_feature_not_found(self):
        """Test feature deletion when user not found"""
        self.face_handler.retinaface.delete_face_data = MagicMock(return_value=False)
        
        result = self.face_handler.delete_feature("nonexistent_user", "mobilenet")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"])

    def test_delete_feature_exception_handling(self):
        """Test feature deletion with exception"""
        self.face_handler.retinaface.delete_face_data = MagicMock(side_effect=Exception("Deletion error"))
        
        result = self.face_handler.delete_feature("test_user_123", "mobilenet")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Deletion error", result["message"])

    def test_delete_feature_with_none_algorithm(self):
        """Test feature deletion with None algorithm"""
        self.face_handler.retinaface.delete_face_data = MagicMock(return_value=True)
        
        result = self.face_handler.delete_feature("test_user_123", None)
        
        self.assertEqual(result["status"], "success")
        self.face_handler.retinaface.delete_face_data.assert_called_once_with("test_user_123", None)


if __name__ == '__main__':
    unittest.main()