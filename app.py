from tokenize import String
from flask import Flask, jsonify, abort, make_response, request
from pip._vendor.appdirs import unicode
import numpy as np
import base64
import json
from io import BytesIO
import cv2, io
from PIL import Image   
import os
from retinaface import Retinaface

app = Flask(__name__)



@app.route('/face/get-list', methods=['GET'])
def get_names():
    known_face_names = np.load("model_data/{backbone}_names.npy".format(backbone="mobilenet"))
    print(known_face_names)
    return jsonify({'names': list(known_face_names)})

@app.route('/face/add-identity', methods=['POST'])
def add_identity(): 
    try:
        data = request.json
        name = (data.get("name"))
        image = (data.get("image"))
        
        # print(image)
        print(name)
        filename = "face_dataset/" + name + "_1.jpg"
        if (image!=None):
            imgdata = base64.b64decode(image)
            img = Image.open(io.BytesIO(imgdata))
            img.save(filename, format="JPEG")

        print("Convert successfully")

        retinaface = Retinaface(1)

        list_dir = os.listdir("face_dataset")
        image_paths = []
        names = []
        for name in list_dir:
            image_paths.append("face_dataset/"+name)
            names.append(name.split("_")[0])

        retinaface.encode_face_dataset(image_paths,names)

        code = "0000"

    except:

        code = "1111"

    print(code)

    return jsonify({'code': code})

@app.route('/face/recognize', methods=['POST']) 
def predict():
    try:
        retinaface = Retinaface()
        data = request.json
        response = {}
        image = (data.get("image"))
        if (image!=None):
            imgdata = base64.b64decode(image)
            img = Image.open(io.BytesIO(imgdata))
            image  = cv2.cvtColor(np.array(img),cv2.COLOR_BGR2RGB)

        r_image, data = retinaface.detect_image(image)

        cv2.imwrite("face.jpg", r_image)
        with open("face.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
       
        print(data)
        response["code"] = "0000"
        response['confidence'] = data
        response['raw_image'] = encoded_string
    except:
        response["code"] = "1111"
        response['confidence'] = "NONE"
        response['raw_image'] = "image error"

    return response
    
if __name__ == '__main__':
    #app.run(debug=True, host="0.0.0.0")
    app.run(host="0.0.0.0", port=5000)
