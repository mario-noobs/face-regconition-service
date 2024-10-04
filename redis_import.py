import numpy as np
import redis
import json

# Assuming `encodings` and `names` are your numpy arrays
encodings = np.load("model_data/{}_face_encoding.npy".format("mobilenet"))
names = np.load("model_data/{}_names.npy".format("mobilenet"))

# Create a dictionary
data_dict = {name: encoding.tolist() for name, encoding in zip(names, encodings)}

# Connect to Redis
# Replace with your Redis server's host, port, and password
redis_host = '75.119.149.223'  # or your Redis server IP
redis_port = 6379          # default Redis port
redis_password = '@2Vietnam'  # your Redis password

# Create a Redis client instance
try:
    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True  # Optional: helps decode bytes to string
    )
    
    # Test the connection
    r.ping()  # This will return True if the connection is successful
    # print("Connected to Redis")

except redis.exceptions.ConnectionError:
    print("Failed to connect to Redis. Check your settings.")

# Store each encoding in a hash
for name, encoding in zip(names, encodings):
    r.hset("mobilenet_face_data", name, encoding.tobytes())