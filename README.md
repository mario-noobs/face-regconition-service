# Face Recognition Service

The Face Recognition Service is a microservice that provides facial recognition functionalities, including creating a new identity and recognizing faces in images. It exposes two main API endpoints: one for creating a new identity and another for recognizing faces from provided images.

## APIs

### 1. **Create Identity**

This API allows you to create a new face identity. You need to provide the image data as a base64 encoded string along with the user ID and request ID. The service will register the identity using the specified algorithms for detection and recognition.

#### Endpoint

```
POST /face/create-identity
```

#### Request Example

```bash
curl --location 'http://localhost:5000/face/create-identity' \
--header 'Content-Type: application/json' \
--data '{
    "algorithmDet": "mobilenet",
    "algorithmReg": "mobilenet",
    "userId": "Mario",
    "requestId": "123",
    "imageBase64": ""
}'
```

#### Request Body

- `algorithmDet` (string): The face detection algorithm to be used (e.g., "mobilenet").
- `algorithmReg` (string): The face recognition algorithm to be used (e.g., "mobilenet").
- `userId` (string): The user ID associated with the face identity.
- `requestId` (string): A unique request ID to track the request.
- `imageBase64` (string): The base64 encoded image data of the user's face.

#### Response

The response will confirm whether the identity creation was successful.

```json
{
    "status": "success",
    "message": "Identity created successfully"
}
```

---

### 2. **Recognize Face**

This API allows you to recognize a face from an image. It will compare the provided image with the registered identities and return the result of the recognition.

#### Endpoint

```
POST /face/recognize
```

#### Request Example

```bash
curl --location 'http://localhost:5000/face/recognize' \
--header 'Content-Type: application/json' \
--data '{
    "algorithmDet": "mobilenet",
    "algorithmReg": "mobilenet",
    "userId": "Mario",
    "requestId": "123",
    "imageBase64": ""
}'
```

#### Request Body

- `algorithmDet` (string): The face detection algorithm to be used (e.g., "mobilenet").
- `algorithmReg` (string): The face recognition algorithm to be used (e.g., "mobilenet").
- `userId` (string): The user ID of the person to recognize.
- `requestId` (string): A unique request ID to track the request.
- `imageBase64` (string): The base64 encoded image data of the face to be recognized.

#### Response

The response will include the recognition result, including whether the face was recognized and the matching identity. 

```json
{
    "code": "0000",
    "data": {
        "request_id": "123",
        "searh_result": {
            "Unknown": "0.9998"
        },
        "user_id": "Mario"
    },
    "message": "Operation successful."
}
```

#### Response Fields

- `code`: A status code indicating the operation's success or failure (e.g., "0000" for success).
- `message`: A human-readable message providing further details about the result.
- `data`:
  - `request_id`: The unique request ID associated with the recognition request.
  - `searh_result`: The result of the face recognition (e.g., the similarity score with known identities). 
    - `"Unknown"`: The score indicating the match with the "Unknown" face.
  - `user_id`: The user ID of the recognized person (if a match is found).

---

## Installation & Setup

### Prerequisites

Ensure that the following software is installed:

- Docker
- Docker Compose

### Setup Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/mario-noobs/face-microservice.git
   cd face-microservice/face-regconition-service
   ```

2. Build the service:

   ```bash
   docker build -t face-recognition-service .
   ```

3. Run the service:

   ```bash
   docker run -p 5000:5000 face-recognition-service
   ```

4. The service will be running at `http://localhost:5000`.

---

## Notes

- **Image Base64 Encoding**: For the image input in the API requests, you must provide the image as a base64 encoded string. Tools like [base64-image.de](https://www.base64-image.de/) can help with encoding images into base64 format.
- **Algorithms**: The service supports various algorithms for face detection and recognition. In this example, `mobilenet` is used for both detection and recognition.
  
---

## Troubleshooting

- If the service is not responding as expected, check the logs for errors:

  ```bash
  docker logs <container-name>
  ```

- Ensure that the base64 image data is correctly formatted and not empty.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
