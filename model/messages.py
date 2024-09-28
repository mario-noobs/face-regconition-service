class Messages:
    SUCCESS = {
        "code": "0000",
        "message": "Operation successful."
    }
    
    MISSING_FIELDS = {
        "code": "4000",
        "message": "Name or image is missing."
    }

    MISSING_ALG = {
        "code": "4002",
        "message": "Algorithm is missing."
    }

    IMAGE_BASE64_ERROR = {
        "code": "4001",
        "message": "Failed to decode the image."
    }
    
    GENERIC_ERROR = {
        "code": "1111",
        "message": "An error occurred while processing the request."
    }
    
    IMAGE_SAVE_ERROR = {
        "code": "5000",
        "message": "Failed to save the image."
    }
    
    IMAGE_ENCODING_ERROR = {
        "code": "5001",
        "message": "Error create image feature"
    }

    NO_FACE = {
        "code": "5002",
        "message": "No face detected"
    }
