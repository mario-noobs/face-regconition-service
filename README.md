# Face Recognition Service

The Face Recognition Service is a microservice that provides facial recognition functionalities, including creating a new identity, recognizing faces in images, and deleting existing identities. It exposes three main API endpoints: one for creating a new identity, another for recognizing faces from provided images, and a third for deleting face identities from the system.

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

### 3. **Delete Identity**

This API allows you to delete an existing face identity from the system. Once deleted, the face will no longer be recognized in future recognition requests.

#### Endpoint

```
DELETE /face/delete-identity
```

#### Request Example

```bash
curl --location --request DELETE 'http://localhost:5000/face/delete-identity' \
--header 'Content-Type: application/json' \
--data '{
    "userId": "Mario",
    "algorithm": "mobilenet",
    "requestId": "125"
}'
```

#### Request Body

- `userId` (string, required): The user ID of the face identity to be deleted.
- `algorithm` (string, optional): The algorithm used for the face encoding. Defaults to "mobilenet" if not specified.
- `requestId` (string, optional): A unique request ID to track the request.

#### Response

The response will confirm whether the identity deletion was successful.

**Success Response (200):**
```json
{
    "status": "success",
    "message": "Face feature deleted successfully",
    "user_id": "Mario",
    "request_id": "125"
}
```

**Error Response (404 - Identity not found):**
```json
{
    "status": "error",
    "message": "Face feature not found for user_id: Mario",
    "user_id": "Mario",
    "request_id": "125"
}
```

**Error Response (400 - Missing userId):**
```json
{
    "status": "error",
    "message": "userId is required",
    "request_id": "125"
}
```

#### Response Fields

- `status`: Indicates whether the operation was successful ("success") or failed ("error").
- `message`: A human-readable message providing details about the result.
- `user_id`: The user ID that was processed.
- `request_id`: The unique request ID associated with the deletion request (if provided).

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

## API Error Codes

The service uses standardized error codes for consistent error handling:

| Code | Message | Description |
|------|---------|-------------|
| 0000 | Operation successful | Success |
| 4000 | Name or image is missing | Missing required fields |
| 4001 | Failed to decode the image | Invalid base64 image |
| 4002 | Algorithm is missing | Missing algorithm parameter |
| 1111 | An error occurred while processing the request | Generic error |
| 5000 | Failed to save the image | Image save error |
| 5001 | Error create image feature | Feature extraction error |
| 5002 | No face detected | No face found in image |

## Notes

- **Image Base64 Encoding**: For the image input in the API requests, you must provide the image as a base64 encoded string. Tools like [base64-image.de](https://www.base64-image.de/) can help with encoding images into base64 format.
- **Algorithms**: The service supports various algorithms for face detection and recognition. Available options include:
  - `mobilenet` (default, recommended for most use cases)
  - `inception_resnetv1` (higher accuracy, more resource intensive)
- **Data Storage**: Face encodings are stored in Redis using a hash structure with keys like `{algorithm}_face_data`.
- **Face Detection**: The service automatically detects faces in provided images and extracts the largest face for processing.
  
---

## API Usage Flow

### Typical Workflow

1. **Create Identity**: Register a new face identity using the `/face/create-identity` endpoint
2. **Recognize Face**: Use the `/face/recognize` endpoint to identify faces in new images
3. **Delete Identity**: Remove unwanted identities using the `/face/delete-identity` endpoint

### Example Complete Flow

```bash
# 1. Create a new identity
curl --location 'http://localhost:5000/face/create-identity' \
--header 'Content-Type: application/json' \
--data '{
    "algorithmDet": "mobilenet",
    "algorithmReg": "mobilenet",
    "userId": "john_doe",
    "requestId": "req_001",
    "imageBase64": "your_base64_image_here"
}'

# 2. Recognize a face
curl --location 'http://localhost:5000/face/recognize' \
--header 'Content-Type: application/json' \
--data '{
    "algorithmDet": "mobilenet",
    "algorithmReg": "mobilenet",
    "userId": "john_doe",
    "requestId": "req_002",
    "imageBase64": "your_base64_image_here"
}'

# 3. Delete the identity when no longer needed
curl --location --request DELETE 'http://localhost:5000/face/delete-identity' \
--header 'Content-Type: application/json' \
--data '{
    "userId": "john_doe",
    "algorithm": "mobilenet",
    "requestId": "req_003"
}'
```

## Troubleshooting

- If the service is not responding as expected, check the logs for errors:

  ```bash
  docker logs <container-name>
  ```

- Ensure that the base64 image data is correctly formatted and not empty.
- Make sure Redis is running and accessible if running the service locally.
- Verify that the provided image contains a clear, front-facing face for better detection accuracy.
- Check that the `userId` exists when performing recognition or deletion operations.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
