import numpy as np
import redis

class RedisStorage:
    def __init__(self, host='75.119.149.223', port=6379, redis_password='@2Vietnam', db=0):
        """Initialize the RedisStorage class and connect to Redis."""
        self.redis_client = redis.Redis(host=host, port=port, 
        password=redis_password, db=db)

    def store_face_data(self, names, encodings, backbone):
        """Store face encodings and names in Redis."""
        for name, encoding in zip(names, encodings):
            self.redis_client.hset(backbone + "_face_data", name, encoding.tobytes())

    def load_face_data(self, backbone):
        """Load face encodings and names from Redis."""
        self.known_face_names = []
        self.known_face_encodings = []
        print(backbone + "_face_data")
        face_data = self.redis_client.hgetall(backbone + "_face_data")
        
        for name, encoding_bytes in face_data.items():
            self.known_face_names.append(name.decode('utf-8'))
            self.known_face_encodings.append(np.frombuffer(encoding_bytes, dtype=np.float32))

        self.known_face_encodings = np.array(self.known_face_encodings)

    def clear_face_data(self, backbone):
        """Clear all face data stored in Redis."""
        self.redis_client.delete(backbone + "_face_data")

