import numpy as np
import redis
import json

# Assuming `encodings` and `names` are your numpy arrays
encodings = np.load("model_data/{}_face_encoding.npy".format("mobilenet"))
names = np.load("model_data/{}_names.npy".format("mobilenet"))

# Create a dictionary
data_dict = {name: encoding.tolist() for name, encoding in zip(names, encodings)}

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Store each encoding in a hash
for name, encoding in zip(names, encodings):
    r.hset("mobilenet_face_data", name, encoding.tobytes())