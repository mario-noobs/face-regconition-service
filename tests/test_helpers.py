import unittest
import base64
import io
import numpy as np
import cv2
import sys
import os
from unittest.mock import patch, MagicMock
from PIL import Image

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    base64_to_numpy_image,
    numpy_image_to_base64,
    is_valid_base64_image,
    convert_string_to_hash,
    get_image_paths_and_names
)


class TestHelpers(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple 3x3 RGB test image
        self.test_image = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
            [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
        ], dtype=np.uint8)
        
        # Create a valid base64 image (1x1 pixel PNG)
        self.valid_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        # Invalid base64 strings
        self.invalid_base64_strings = [
            "invalid_base64",
            "not@base64!",
            "",
            "iVBORw0KGgoAAAANSUhEUgAA",  # Incomplete base64
        ]

    def test_base64_to_numpy_image_valid(self):
        """Test conversion of valid base64 image to numpy array"""
        # First convert numpy image to base64, then back to numpy
        _, buffer = cv2.imencode('.jpg', self.test_image)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        result = base64_to_numpy_image(base64_string)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 3)  # Should be 3D array (H, W, C)
        self.assertEqual(result.shape[2], 3)    # Should have 3 channels (RGB/BGR)

    def test_base64_to_numpy_image_invalid(self):
        """Test conversion of invalid base64 string"""
        with self.assertRaises(Exception):
            base64_to_numpy_image("invalid_base64_string")

    def test_numpy_image_to_base64_valid(self):
        """Test conversion of numpy image to base64"""
        result = numpy_image_to_base64(self.test_image)
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        
        # Verify it's valid base64
        try:
            decoded = base64.b64decode(result)
            self.assertTrue(len(decoded) > 0)
        except Exception:
            self.fail("Generated base64 string is invalid")

    def test_numpy_image_to_base64_roundtrip(self):
        """Test roundtrip conversion: numpy -> base64 -> numpy"""
        # Convert to base64
        base64_string = numpy_image_to_base64(self.test_image)
        
        # Convert back to numpy
        reconstructed_image = base64_to_numpy_image(base64_string)
        
        self.assertIsInstance(reconstructed_image, np.ndarray)
        # Note: Due to JPEG compression, exact pixel values might differ
        # so we just check dimensions
        self.assertEqual(reconstructed_image.shape[2], 3)

    def test_is_valid_base64_image_valid(self):
        """Test validation of valid base64 image"""
        result = is_valid_base64_image(self.valid_base64_image)
        self.assertTrue(result)

    def test_is_valid_base64_image_invalid(self):
        """Test validation of invalid base64 strings"""
        for invalid_string in self.invalid_base64_strings:
            with self.subTest(invalid_string=invalid_string):
                result = is_valid_base64_image(invalid_string)
                self.assertFalse(result)

    def test_is_valid_base64_image_valid_base64_but_not_image(self):
        """Test validation of valid base64 but not an image"""
        # Valid base64 but not an image
        not_image_base64 = base64.b64encode(b"This is not an image").decode('utf-8')
        result = is_valid_base64_image(not_image_base64)
        self.assertFalse(result)

    def test_convert_string_to_hash_consistent(self):
        """Test that string hashing is consistent"""
        test_string = "test_string_123"
        
        hash1 = convert_string_to_hash(test_string)
        hash2 = convert_string_to_hash(test_string)
        
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, bytes)
        self.assertTrue(len(hash1) > 0)

    def test_convert_string_to_hash_different_inputs(self):
        """Test that different strings produce different hashes"""
        string1 = "string1"
        string2 = "string2"
        
        hash1 = convert_string_to_hash(string1)
        hash2 = convert_string_to_hash(string2)
        
        self.assertNotEqual(hash1, hash2)

    def test_convert_string_to_hash_empty_string(self):
        """Test hashing of empty string"""
        result = convert_string_to_hash("")
        
        self.assertIsInstance(result, bytes)
        self.assertTrue(len(result) > 0)

    def test_convert_string_to_hash_unicode(self):
        """Test hashing of unicode string"""
        unicode_string = "测试字符串🎉"
        
        result = convert_string_to_hash(unicode_string)
        
        self.assertIsInstance(result, bytes)
        self.assertTrue(len(result) > 0)

    @patch('os.listdir')
    def test_get_image_paths_and_names_success(self, mock_listdir):
        """Test successful retrieval of image paths and names"""
        # Mock directory listing
        mock_listdir.return_value = [
            "user1_001.jpg",
            "user2_002.png",
            "user3_003.jpeg"
        ]
        
        dataset_dir = "/fake/dataset/dir"
        image_paths, names = get_image_paths_and_names(dataset_dir)
        
        expected_paths = [
            "/fake/dataset/dir/user1_001.jpg",
            "/fake/dataset/dir/user2_002.png", 
            "/fake/dataset/dir/user3_003.jpeg"
        ]
        expected_names = ["user1", "user2", "user3"]
        
        self.assertEqual(image_paths, expected_paths)
        self.assertEqual(names, expected_names)
        mock_listdir.assert_called_once_with(dataset_dir)

    @patch('os.listdir')
    def test_get_image_paths_and_names_empty_directory(self, mock_listdir):
        """Test retrieval from empty directory"""
        mock_listdir.return_value = []
        
        dataset_dir = "/empty/dir"
        image_paths, names = get_image_paths_and_names(dataset_dir)
        
        self.assertEqual(image_paths, [])
        self.assertEqual(names, [])

    @patch('os.listdir')
    def test_get_image_paths_and_names_complex_filenames(self, mock_listdir):
        """Test retrieval with complex filenames"""
        mock_listdir.return_value = [
            "john_doe_001_profile.jpg",
            "jane_smith_002_front.png",
            "bob_jones_003.jpeg"
        ]
        
        dataset_dir = "/complex/dir"
        image_paths, names = get_image_paths_and_names(dataset_dir)
        
        expected_names = ["john", "jane", "bob"]  # First part before underscore
        
        self.assertEqual(len(image_paths), 3)
        self.assertEqual(names, expected_names)

    @patch('os.listdir')
    def test_get_image_paths_and_names_exception(self, mock_listdir):
        """Test exception handling in get_image_paths_and_names"""
        mock_listdir.side_effect = OSError("Directory not found")
        
        dataset_dir = "/nonexistent/dir"
        
        with self.assertRaises(OSError):
            get_image_paths_and_names(dataset_dir)

    def test_base64_to_numpy_image_grayscale(self):
        """Test conversion with grayscale image"""
        # Create a simple grayscale test image
        gray_image = np.array([[128, 64], [192, 32]], dtype=np.uint8)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', gray_image)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        # Convert back
        result = base64_to_numpy_image(base64_string)
        
        self.assertIsInstance(result, np.ndarray)
        # OpenCV might convert grayscale to 3-channel when decoding
        self.assertTrue(len(result.shape) >= 2)

    def test_numpy_image_to_base64_different_formats(self):
        """Test base64 conversion with different image formats"""
        # Test with different numpy array shapes and types
        test_images = [
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8),  # RGB
            np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8),  # Different dimensions
        ]
        
        for i, test_img in enumerate(test_images):
            with self.subTest(image_index=i):
                result = numpy_image_to_base64(test_img)
                
                self.assertIsInstance(result, str)
                self.assertTrue(len(result) > 0)
                
                # Verify it's valid base64
                try:
                    base64.b64decode(result)
                except Exception:
                    self.fail(f"Generated base64 string is invalid for test image {i}")


if __name__ == '__main__':
    unittest.main()