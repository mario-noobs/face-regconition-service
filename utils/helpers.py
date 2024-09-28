import os
import io
import base64
import logging
from flask import Flask, request, jsonify
from PIL import Image
import re
import hashlib
import base64

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

def is_base64_image(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s
    except Exception:
        return False

def convert_string_to_hash(word):
    digest = hashlib.sha1(word.encode('utf-16-le')).digest()
    return base64.b64encode(digest)