from pydantic import BaseModel


class PredictionResponse(BaseModel):
    pretrained_model: str
    top_similar_images: list


class ErrorResponse(BaseModel):
    error: str


RESPONSE_MODELS = {
    200: {
        "model": PredictionResponse,
        "description": "Successfully processed the image and retrieved similar images.",
        "content": {
            "application/json": {
                "example": {
                    "pretrained_model": "vgg16",
                    "top_4_similar_images": [
                        {
                            "id": 633755,
                            "name": "3694014992_3035f3d421_o.jpg",
                            "score": 0.27253270149230957,
                        },
                        {
                            "id": 67089,
                            "name": "187004065_b3d8f4666c_o.jpg",
                            "score": 0.2689032554626465,
                        },
                    ],
                }
            }
        },
    },
    400: {
        "model": ErrorResponse,
        "description": "Invalid input or could not preprocess the image.",
        "content": {
            "application/json": {
                "example": {"error": "Could not preprocess the image."}
            }
        },
    },
    500: {
        "model": ErrorResponse,
        "description": "An unexpected error occurred.",
        "content": {
            "application/json": {"detail": {"error": "An unexpected error occurred."}}
        },
    },
}
