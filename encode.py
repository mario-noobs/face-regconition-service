import base64

with open("/Users/standardlab/Downloads/Photo 2022-09-15 11.50.54 AM.HEIF", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    
print(encoded_string)

