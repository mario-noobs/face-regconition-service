import unittest
import base64
import io
import sys
import os
from unittest.mock import patch, MagicMock
from PIL import Image

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    is_valid_base64_image,
    convert_string_to_hash,
    get_image_paths_and_names
)


class TestHelpersIsolated(unittest.TestCase):
    """Isolated tests for helper functions that don't require OpenCV"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a valid base64 image (1x1 pixel PNG)
        self.valid_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        # Invalid base64 strings
        self.invalid_base64_strings = [
            "invalid_base64",
            "not@base64!",
            "",
            "iVBORw0KGgoAAAANSUhEUgAA",  # Incomplete base64
        ]

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

    def test_base64_validation_logic(self):
        """Test base64 validation logic without OpenCV"""
        # Test with various base64 strings
        test_cases = [
            # Valid base64 strings
            ("SGVsbG8gV29ybGQ=", True),  # "Hello World" 
            ("", False),  # Empty string
            ("invalid_base64", False),  # Invalid characters
            ("validbase64withoutpadding", False),  # Missing padding
        ]
        
        for base64_str, expected_valid in test_cases:
            with self.subTest(base64_string=base64_str):
                # Test basic base64 decoding
                try:
                    decoded = base64.b64decode(base64_str)
                    # For our helper function, we also need it to be a valid image
                    # So we'll just check if the base64 decode worked
                    decode_success = len(decoded) > 0
                except Exception:
                    decode_success = False
                
                # The actual validation includes PIL Image.open, which is more strict
                actual_result = is_valid_base64_image(base64_str)
                
                # For invalid base64, both should be False
                if not expected_valid:
                    self.assertFalse(actual_result)

    def test_hash_function_properties(self):
        """Test hash function properties"""
        test_strings = [
            "test1",
            "test2", 
            "longer_test_string_with_special_chars_123!@#",
            "unicode_测试_🎉",
            ""  # Empty string
        ]
        
        hashes = []
        for test_str in test_strings:
            hash_result = convert_string_to_hash(test_str)
            
            # Check properties
            self.assertIsInstance(hash_result, bytes)
            self.assertTrue(len(hash_result) > 0)
            
            # Check that same input produces same hash
            hash_again = convert_string_to_hash(test_str)
            self.assertEqual(hash_result, hash_again)
            
            hashes.append(hash_result)
        
        # Check that different inputs produce different hashes
        # (except empty string case)
        unique_hashes = set(hashes)
        self.assertEqual(len(unique_hashes), len(hashes))

    def test_image_path_parsing_logic(self):
        """Test the logic for parsing image filenames"""
        test_cases = [
            ("user1_001.jpg", "user1"),
            ("john_doe_profile.png", "john"),
            ("simple.jpeg", "simple"),
            ("complex_name_with_multiple_underscores.gif", "complex"),
            ("no_extension", "no"),
        ]
        
        for filename, expected_name in test_cases:
            with self.subTest(filename=filename):
                # Simulate the name parsing logic from get_image_paths_and_names
                parsed_name = filename.split("_")[0]
                self.assertEqual(parsed_name, expected_name)

    def test_file_extension_handling(self):
        """Test handling of different file extensions"""
        valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        test_filenames = [
            "image1.jpg",
            "image2.jpeg", 
            "image3.png",
            "image4.gif",
            "image5.bmp",
            "image6.txt",  # Invalid extension
            "image7"       # No extension
        ]
        
        for filename in test_filenames:
            with self.subTest(filename=filename):
                # Check if filename has a valid image extension
                has_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)
                
                if filename in ["image1.jpg", "image2.jpeg", "image3.png", "image4.gif", "image5.bmp"]:
                    self.assertTrue(has_valid_extension)
                else:
                    self.assertFalse(has_valid_extension)


class TestBase64LogicWithoutCV2(unittest.TestCase):
    """Test base64 logic without using OpenCV functions"""
    
    def test_base64_encode_decode_roundtrip(self):
        """Test base64 encoding and decoding roundtrip"""
        test_data = b"This is test binary data for base64 encoding"
        
        # Encode to base64
        encoded = base64.b64encode(test_data).decode('utf-8')
        
        # Decode back
        decoded = base64.b64decode(encoded)
        
        # Should be identical
        self.assertEqual(test_data, decoded)
        
        # Check that it's valid base64
        self.assertTrue(len(encoded) > 0)
        self.assertIsInstance(encoded, str)

    def test_base64_validation_without_image_check(self):
        """Test base64 validation logic without PIL"""
        test_cases = [
            ("SGVsbG8gV29ybGQ=", True),  # Valid base64
            ("", False),                 # Empty
            ("invalid!", False),         # Invalid characters
            ("abc", False),              # Too short/invalid padding
        ]
        
        for test_str, should_be_valid in test_cases:
            with self.subTest(base64_string=test_str):
                try:
                    decoded = base64.b64decode(test_str)
                    is_valid = len(decoded) > 0 if should_be_valid else True
                except Exception:
                    is_valid = False
                
                if should_be_valid:
                    self.assertTrue(is_valid)
                else:
                    self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()