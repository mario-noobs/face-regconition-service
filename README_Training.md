## Facenet + Retinaface: Implementation of Face Recognition Models in PyTorch
---

## Table of Contents
1. [Attention](#attention)
2. [Environment](#environment)
3. [Download](#download)
4. [Prediction Steps](#prediction-steps)
5. [References](#references)

## Attention
This library contains two networks: Retinaface and Facenet, which use different weights.  
Be sure to pay attention to the selection of weights and the matching of backbones with weights.  
The Retinaface face detection repository can be trained and used for prediction: [Retinaface-PyTorch](https://github.com/bubbliiiing/retinaface-pytorch)  
The Facenet face recognition repository can also be trained and used for prediction: [Facenet-PyTorch](https://github.com/bubbliiiing/facenet-pytorch)  

## Environment
`pytorch==1.2.0`

## Download
The weight files needed for prediction can be downloaded from Baidu Cloud.  
Link: [https://pan.baidu.com/s/1iTo4_x0DHg0GoTUQWduMZw](https://pan.baidu.com/s/1iTo4_x0DHg0GoTUQWduMZw) Extraction Code: `dmw6`  

## Prediction Steps
1. This project comes with Retinaface and Facenet models based on MobileNet. You can run them directly. If you want to use the Retinaface model based on ResNet50 and the Facenet model based on Inception ResNet v1, you need to modify the configurations.
2. In the `retinaface.py` file, modify the `model_path` and `backbone` in the following section to match the trained files:  
    ```python
    _defaults = {
        "retinaface_model_path" : 'model_data/Retinaface_mobilenet0.25.pth',
        #-----------------------------------#
        #   Optional retinaface_backbone options:
        #   mobilenet and resnet50
        #-----------------------------------#
        "retinaface_backbone"   : "mobilenet",
        "confidence"            : 0.5,
        "iou"                   : 0.3,
        #----------------------------------------------------------------------#
        #   Whether to limit the image size.
        #   Input image size significantly affects FPS. To speed up detection, 
        #   you can reduce input_shape. 
        #   If enabled, the input image size will be limited to input_shape; 
        #   otherwise, the original image will be used for prediction.
        #   This may lead to deviations in detection results; this issue does 
        #   not exist with ResNet50.
        #   Adjust input_shape based on the size of the input image, making 
        #   sure it's a multiple of 32, e.g., [640, 640, 3].
        #----------------------------------------------------------------------#
        "retinaface_input_shape": [640, 640, 3],
        #-----------------------------------#
        #   Whether to limit the image size
        #-----------------------------------#
        "letterbox_image"       : True,
        
        "facenet_model_path"    : 'facenet_inception_resnetv1.pth',
        #-----------------------------------#
        #   Optional facenet_backbone options:
        #   mobilenet and inception_resnetv1
        #-----------------------------------#
        "facenet_backbone"      : "inception_resnetv1",
        "facenet_input_shape"   : [160,160,3],
        "facenet_threshold"      : 0.9,

        "cuda"                  : True
    }
    ```
3. Run `encoding.py` to encode the images in the `face_dataset` folder. The naming convention for images is `XXX_1.jpg`, `XXX_2.jpg`. The corresponding database face encoding data file will be generated in the `model_data` folder.
4. Run `predict.py` and enter the following text to make a prediction:  
    ```python
    img/zhangxueyou.jpg
    ```  
5. In `predict.py`, settings can be adjusted for FPS testing and video detection.  

## References
[https://github.com/biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

