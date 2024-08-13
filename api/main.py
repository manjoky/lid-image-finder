import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "utils"))
)

from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from utils import preprocess, models
from pydantic import BaseModel, validator
from utils.logger import logger
from api import response_examples as re

app = FastAPI()


class PredictRequest(BaseModel):
    model_name: str = Form("resnet50")
    nb_similar_images: int = Form(3)
    retrieved_ref: str = Form("id,name,score")
    ignore_duplicate: bool = Form(True)

    @validator("retrieved_ref")
    def validate_retrieved_ref(cls, v):
        allowed_fields = {"id", "name", "score"}
        cls.requested_fields = {field.strip() for field in v.split(",")}
        invalid_fields = cls.requested_fields - allowed_fields
        if invalid_fields:
            raise ValueError(
                f"Invalid fields in 'retrieved_ref': {', '.join(invalid_fields)}. Allowed fields are: {', '.join(allowed_fields)}"
            )
        return v

    @validator("model_name")
    def validate_model_name(cls, v):
        allowed_models = {"resnet50", "vgg16"}
        if v not in allowed_models:
            raise ValueError(
                f"Invalid model_name '{v}'. Allowed values are: {', '.join(allowed_models)}"
            )
        return v

@app.post("/predict/", response_model=re.PredictionResponse, responses=re.RESPONSE_MODELS)
async def predict(
    file: UploadFile,
    model_name: str = Form("resnet50", description="Name of the pre-trained model for image comparison. Possible options: `resnet50`, `vgg16`."),
    nb_similar_images: int = Form(3, description="Number of similar images to return."),
    retrieved_ref: str = Form("id,name,score", description="Attributes to be returned for each similar image. Possible options : `id`, `name`, `score`."),
    ignore_duplicate: bool = Form(True, description="Ignore the duplicate input image in the results."),
):
    """
    Receives an image file and prediction parameters to return similar images based on the chosen model.
    
    **Responses:**
    - **200 OK**: Returns a JSON object with the prediction results.
    - **400 Bad Request**: If the image preprocessing fails or if any parameters are invalid.
    - **500 Internal Server Error**: For unexpected errors during the processing.
    """
    try:
        form_data = PredictRequest(
            model_name=model_name,
            nb_similar_images=nb_similar_images,
            retrieved_ref=retrieved_ref,
            ignore_duplicate=ignore_duplicate,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    logger.info(f"Received request to /predict/")
    try:
        # Load and pre-process the input image
        image = preprocess.preprocess_image_from_file(file.file)
        if image is None:
            logger.error("Could not preprocess the image.")
            return JSONResponse(
                content={"error": "Could not preprocess the image."}, status_code=400
            )

        top_nb = models.predict(
            image_name=file.filename,
            image_tensor=image,
            model_name=form_data.model_name,
            ignore_duplicate=form_data.ignore_duplicate,
            nb_similar_images=form_data.nb_similar_images,
        )

        result = [{k: v for k, v in elem.items() if k in form_data.requested_fields} for elem in top_nb]

        logger.info(
            f"Successfully processed the image and retrieved top {form_data.nb_similar_images} similar images."
        )
        # Return the N closest images
        return JSONResponse(
            content={"pretrained_model": form_data.model_name, f"top_{form_data.nb_similar_images}_similar_images": result}
        )

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            detail={"error": f"An unexpected error occurred.\n{e}"}, status_code=500
        )
