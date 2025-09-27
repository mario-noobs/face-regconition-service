import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Redis before importing modules that use it
sys.modules['redis'] = MagicMock()

from databases.redis_db import RedisStorage


class TestRedisStorage(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Mock Redis client
        self.mock_redis = MagicMock()
        
        # Create RedisStorage instance with mocked Redis
        with patch('databases.redis_db.redis.Redis', return_value=self.mock_redis):
            self.redis_storage = RedisStorage()
        
        self.test_backbone = "mobilenet"
        self.test_names = ["user1", "user2"]
        self.test_encodings = [np.random.rand(512).astype(np.float32), 
                              np.random.rand(512).astype(np.float32)]

    def test_redis_storage_initialization(self):
        """Test RedisStorage initialization"""
        with patch('databases.redis_db.redis.Redis') as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_class.return_value = mock_redis_instance
            
            storage = RedisStorage(host='localhost', port=6379, db=0)
            
            mock_redis_class.assert_called_once_with(
                host='localhost',
                port=6379,
                password=None,
                db=0
            )

    def test_store_face_data_success(self):
        """Test successful face data storage"""
        self.redis_storage.store_face_data(self.test_names, self.test_encodings, self.test_backbone)
        
        # Verify hset was called for each name/encoding pair
        self.assertEqual(self.mock_redis.hset.call_count, len(self.test_names))
        
        # Check the calls
        calls = self.mock_redis.hset.call_args_list
        expected_key = f"{self.test_backbone}_face_data"
        
        for i, call in enumerate(calls):
            args, kwargs = call
            self.assertEqual(args[0], expected_key)  # Redis key
            self.assertEqual(args[1], self.test_names[i])  # Name
            # args[2] should be encoded bytes of the numpy array

    def test_store_face_data_exception(self):
        """Test face data storage with exception"""
        self.mock_redis.hset.side_effect = Exception("Redis connection error")
        
        with self.assertRaises(Exception):
            self.redis_storage.store_face_data(self.test_names, self.test_encodings, self.test_backbone)

    def test_load_face_data_success(self):
        """Test successful face data loading"""
        # Mock Redis response
        mock_face_data = {
            b'user1': np.random.rand(512).astype(np.float32).tobytes(),
            b'user2': np.random.rand(512).astype(np.float32).tobytes()
        }
        self.mock_redis.hgetall.return_value = mock_face_data
        
        self.redis_storage.load_face_data(self.test_backbone)
        
        # Verify the method was called with correct key
        expected_key = f"{self.test_backbone}_face_data"
        self.mock_redis.hgetall.assert_called_once_with(expected_key)
        
        # Check that known_face_names and known_face_encodings are set
        self.assertEqual(len(self.redis_storage.known_face_names), 2)
        self.assertEqual(len(self.redis_storage.known_face_encodings), 2)
        self.assertIn('user1', self.redis_storage.known_face_names)
        self.assertIn('user2', self.redis_storage.known_face_names)

    def test_load_face_data_empty(self):
        """Test face data loading when no data exists"""
        self.mock_redis.hgetall.return_value = {}
        
        self.redis_storage.load_face_data(self.test_backbone)
        
        self.assertEqual(len(self.redis_storage.known_face_names), 0)
        self.assertEqual(len(self.redis_storage.known_face_encodings), 0)

    def test_load_face_data_exception(self):
        """Test face data loading with exception"""
        self.mock_redis.hgetall.side_effect = Exception("Redis connection error")
        
        with self.assertRaises(Exception):
            self.redis_storage.load_face_data(self.test_backbone)

    def test_clear_face_data_success(self):
        """Test successful face data clearing"""
        self.mock_redis.delete.return_value = 1
        
        self.redis_storage.clear_face_data(self.test_backbone)
        
        expected_key = f"{self.test_backbone}_face_data"
        self.mock_redis.delete.assert_called_once_with(expected_key)

    def test_clear_face_data_exception(self):
        """Test face data clearing with exception"""
        self.mock_redis.delete.side_effect = Exception("Redis connection error")
        
        with self.assertRaises(Exception):
            self.redis_storage.clear_face_data(self.test_backbone)

    def test_store_and_load_roundtrip(self):
        """Test storing and loading data roundtrip"""
        # Setup mock for store
        self.mock_redis.hset.return_value = True
        
        # Setup mock for load
        encoding1_bytes = self.test_encodings[0].tobytes()
        encoding2_bytes = self.test_encodings[1].tobytes()
        mock_face_data = {
            b'user1': encoding1_bytes,
            b'user2': encoding2_bytes
        }
        self.mock_redis.hgetall.return_value = mock_face_data
        
        # Store data
        self.redis_storage.store_face_data(self.test_names, self.test_encodings, self.test_backbone)
        
        # Load data
        self.redis_storage.load_face_data(self.test_backbone)
        
        # Verify
        self.assertEqual(len(self.redis_storage.known_face_names), 2)
        self.assertEqual(self.redis_storage.known_face_names, ['user1', 'user2'])
        self.assertEqual(self.redis_storage.known_face_encodings.shape, (2, 512))

    def test_different_backbones(self):
        """Test storage with different backbone algorithms"""
        backbones = ["mobilenet", "inception_resnetv1", "custom_backbone"]
        
        for backbone in backbones:
            with self.subTest(backbone=backbone):
                self.redis_storage.store_face_data(self.test_names, self.test_encodings, backbone)
                
                # Verify correct key was used
                expected_key = f"{backbone}_face_data"
                calls = self.mock_redis.hset.call_args_list
                # Check the most recent calls
                for call in calls[-len(self.test_names):]:
                    args, kwargs = call
                    self.assertEqual(args[0], expected_key)

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        with patch('databases.redis_db.redis.Redis') as mock_redis_class:
            mock_redis_instance = MagicMock()
            mock_redis_class.return_value = mock_redis_instance
            
            storage = RedisStorage(
                host='custom_host',
                port=6380,
                redis_password='secret',
                db=1
            )
            
            mock_redis_class.assert_called_once_with(
                host='custom_host',
                port=6380,
                password='secret',
                db=1
            )


if __name__ == '__main__':
    unittest.main()