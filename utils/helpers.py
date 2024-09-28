import os
import io
import base64
import logging
from flask import Flask, request, jsonify
from PIL import Image
import re
import hashlib
import base64
import numpy as np
import cv2

def base64_to_numpy_image(base64_string):
    """Convert a Base64 string to a NumPy array."""
    # Decode the Base64 string
    image_data = base64.b64decode(base64_string)

    # Convert the byte data to a NumPy array
    np_array = np.frombuffer(image_data, dtype=np.uint8)

    # Decode the image array to an actual image using OpenCV
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return image

def numpy_image_to_base64(image_array):
    """Convert a NumPy array to a Base64 string."""
    # Encode the image to a memory buffer using OpenCV
    _, buffer = cv2.imencode('.jpg', image_array)
    
    # Convert the buffer to bytes and then to a Base64 string
    base64_string = base64.b64encode(buffer).decode('utf-8')
    
    return base64_string

def save_image(image_data, filename):
    """Decodes and saves the image from base64 data."""
    try:
        imgdata = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(imgdata))
        img.save(filename, format=IMAGE_FORMAT)
        logger.info("Image saved successfully: %s", filename)
    except Exception as e:
        logger.error("Failed to save image: %s", e)
        raise

def get_image_paths_and_names(dataset_dir):
    """Retrieves image paths and corresponding names from the dataset directory."""
    image_paths = []
    names = []
    try:
        list_dir = os.listdir(dataset_dir)
        for name in list_dir:
            image_paths.append(os.path.join(dataset_dir, name))
            names.append(name.split("_")[0])
    except Exception as e:
        logger.error("Error reading image paths: %s", e)
        raise
    return image_paths, names

def is_valid_base64_image(image_string):
    # checking valid base64 image string 
    result = True
    try:
        image = base64.b64decode(image_string)
        img = Image.open(io.BytesIO(image))
    except Exception:
        result = False
    
    return result

def convert_string_to_hash(word):
    digest = hashlib.sha1(word.encode('utf-16-le')).digest()
    return base64.b64encode(digest)

