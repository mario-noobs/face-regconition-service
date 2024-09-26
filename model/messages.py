class Messages:
    SUCCESS = {
        "code": "0000",
        "message": "Operation successful."
    }
    
    MISSING_FIELDS = {
        "code": "4000",
        "message": "Name or image is missing."
    }
    
    GENERIC_ERROR = {
        "code": "1111",
        "message": "An error occurred while processing the request."
    }
    
    IMAGE_SAVE_ERROR = {
        "code": "5000",
        "message": "Failed to save the image."
    }
    
    IMAGE_READ_ERROR = {
        "code": "5001",
        "message": "Error reading image paths."
    }
