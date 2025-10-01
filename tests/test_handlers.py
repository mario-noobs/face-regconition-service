import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the heavy dependencies before importing
with patch('retinaface.Retinaface') as mock_retinaface_class:
    mock_retinaface_instance = MagicMock()
    mock_retinaface_class.return_value = mock_retinaface_instance
    
    from handler.handlers import FaceHandler, FaceFeatureExtractionInterface
    from model.request import CreateFaceFeatureRequest, RecognizeFaceFeatureRequest
    from model.response import CreateFaceFeatureResponse, RecognizeFaceFeatureResponse
    from model.exceptions import FaceFeatureException
    from model.messages import Messages


class TestFaceHandler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.mock_logger = MagicMock()
        
        # Create FaceHandler with mocked Retinaface
        with patch('handler.handlers.Retinaface') as mock_retinaface_class:
            self.mock_retinaface = MagicMock()
            mock_retinaface_class.return_value = self.mock_retinaface
            self.face_handler = FaceHandler(logger=self.mock_logger)
        
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

    def test_face_handler_implements_interface(self):
        """Test that FaceHandler implements FaceFeatureExtractionInterface"""
        self.assertIsInstance(self.face_handler, FaceFeatureExtractionInterface)

    def test_create_feature_success(self):
        """Test successful feature creation"""
        # Mock successful encoding result
        mock_encoding_result = {
            "status": 1,
            "message": "Success",
            "face_encoding_base64": "encoded_features",
            "encoding_shape": [1, 512]
        }
        self.mock_retinaface.encode_face_image.return_value = mock_encoding_result
        
        response = self.face_handler.create_feature(self.create_request)
        
        self.assertIsInstance(response, CreateFaceFeatureResponse)
        self.assertEqual(response.request_id, "req_123")
        self.assertEqual(response.user_id, "test_user_123")
        self.assertEqual(response.code, Messages.SUCCESS['code'])
        self.assertEqual(response.message, Messages.SUCCESS['message'])

    # def test_create_feature_no_face_detected(self):
    #     """Test feature creation when no face is detected"""
    #     # Mock no face detected result
    #     mock_encoding_result = {
    #         "status": 0,
    #         "message": "No face detected"
    #     }
    #     self.mock_retinaface.encode_face_image.return_value = mock_encoding_result
        
    #     with self.assertRaises(FaceFeatureException) as context:
    #         self.face_handler.create_feature(self.create_request)
        
    #     self.assertEqual(context.exception.code, Messages.NO_FACE['code'])
    #     self.assertEqual(context.exception.message, Messages.NO_FACE['message'])

    # def test_create_feature_encoding_error(self):
    #     """Test feature creation with encoding error"""
    #     # Mock encoding error result
    #     mock_encoding_result = {
    #         "status": -1,
    #         "message": "Encoding failed"
    #     }
    #     self.mock_retinaface.encode_face_image.return_value = mock_encoding_result
        
    #     with self.assertRaises(FaceFeatureException) as context:
    #         self.face_handler.create_feature(self.create_request)
        
    #     self.assertEqual(context.exception.code, Messages.IMAGE_ENCODING_ERROR['code'])

    def test_create_feature_exception_handling(self):
        """Test feature creation with exception"""
        # Mock exception during encoding
        self.mock_retinaface.encode_face_image.side_effect = Exception("Encoding error")
        
        response = self.face_handler.create_feature(self.create_request)
        
        self.assertIsInstance(response, CreateFaceFeatureResponse)
        self.assertEqual(response.code, Messages.GENERIC_ERROR['code'])
        self.assertIn("Encoding error", response.message)

    def test_face_search_success(self):
        """Test successful face search"""
        # Mock successful search result
        mock_search_result = {
            "status": 1,
            "message": "Match found",
            "search_result": {
                "match_score": 0.95,
                "matched_user": "test_user_123"
            }
        }
        # Mock the actual method calls
        self.mock_retinaface.reload_face_feature.return_value = None
        self.mock_retinaface.search_face.return_value = (True, mock_search_result["search_result"])
        
        response = self.face_handler.face_search(self.recognize_request)
        
        self.assertIsInstance(response, RecognizeFaceFeatureResponse)
        self.assertEqual(response.request_id, "req_456")
        self.assertEqual(response.user_id, "test_user_123")
        self.assertEqual(response.code, Messages.SUCCESS['code'])
        self.assertEqual(response.message, Messages.SUCCESS['message'])

    # def test_face_search_no_face_detected(self):
    #     """Test face search when no face is detected"""
    #     # Mock no face detected result
    #     mock_search_result = {
    #         "status": 0,
    #         "message": "No face detected"
    #     }
    #     # Mock the actual method calls - no data returned
    #     self.mock_retinaface.reload_face_feature.return_value = None
    #     self.mock_retinaface.search_face.return_value = (False, None)
        
    #     with self.assertRaises(FaceFeatureException) as context:
    #         self.face_handler.face_search(self.recognize_request)
        
    #     self.assertEqual(context.exception.code, Messages.NO_FACE['code'])

    def test_face_search_encoding_error(self):
        """Test face search with encoding error"""
        # Mock encoding error result
        mock_search_result = {
            "status": -1,
            "message": "Search failed"
        }
        # Mock the actual method calls - exception during search
        self.mock_retinaface.reload_face_feature.return_value = None
        self.mock_retinaface.search_face.side_effect = Exception("Search failed")
        
        response = self.face_handler.face_search(self.recognize_request)
        
        self.assertEqual(response.code, Messages.GENERIC_ERROR['code'])

    def test_face_search_exception_handling(self):
        """Test face search with exception"""
        # Mock exception during search
        self.mock_retinaface.reload_face_feature.side_effect = Exception("Search error")
        
        response = self.face_handler.face_search(self.recognize_request)
        
        self.assertIsInstance(response, RecognizeFaceFeatureResponse)
        self.assertEqual(response.code, Messages.GENERIC_ERROR['code'])
        self.assertIn("Search error", response.message)

    def test_delete_feature_success(self):
        """Test successful feature deletion"""
        # Mock successful deletion
        self.mock_retinaface.delete_face_data.return_value = True
        
        result = self.face_handler.delete_feature("test_user_123", "mobilenet")
        
        self.assertEqual(result["status"], "success")
        self.mock_retinaface.delete_face_data.assert_called_once_with("test_user_123", "mobilenet")

    def test_delete_feature_not_found(self):
        """Test feature deletion when user not found"""
        # Mock user not found
        self.mock_retinaface.delete_face_data.return_value = False
        
        result = self.face_handler.delete_feature("nonexistent_user", "mobilenet")
        
        self.assertEqual(result["status"], "error")

    def test_delete_feature_exception_handling(self):
        """Test feature deletion with exception"""
        # Mock exception during deletion
        self.mock_retinaface.delete_face_data.side_effect = Exception("Deletion error")
        
        result = self.face_handler.delete_feature("test_user_123", "mobilenet")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Deletion error", result["message"])

    def test_delete_feature_default_algorithm(self):
        """Test feature deletion with default algorithm"""
        # Mock successful deletion
        self.mock_retinaface.delete_face_data.return_value = True
        
        # Test with None algorithm 
        result = self.face_handler.delete_feature("test_user_123", None)
        
        self.assertEqual(result["status"], "success")
        # Verify it was called with the algorithm parameter
        self.mock_retinaface.delete_face_data.assert_called_once_with("test_user_123", None)


if __name__ == '__main__':
    unittest.main()