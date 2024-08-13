## Project Overview

### Quick description
This project is a ML-based image retrieval system. It allows users to upload an image and retrieve the closest images. The system can be used through 2 different solutions:
* **FastAPI API** which implements a POST route to receive an image as input and returns the ids of the n most similar images from the dataset.
* **Gradio App *(Optional)*** which offers a simplified interface to interact with the system and visualize the results.

### Repository Structure
The repository is organized into the following directories:

>* `data`: contains the image dataset, feature databases and the code to generate those key elements
>* `utils`: contains utility functions for image preprocessing, feature extraction and similarity computation
>* `api`: contains the FastAPI API code
>* `app`: contains the Gradio app code
>* `test`: contains the unit tests (not implemented within the timeframe)
>* `explo_analysis.ipynb`: contains a jupyter notebook where we did our EDA

## Using the Code

#### Preliminary steps
* Create and activate a virtual environment --> `python -m venv .venv` and `.\.venv\Scripts\activate`
* Install dependencies --> `pip install -r requirements.txt`
* Download images --> `python .\data\image-downloader.py --csv_file .\data\image_urls.csv`
* Generate featuresDB --> `python .\data\feature_db_creator.py --csv_file .\data\image_urls.csv`

#### API-specific
* Launch the API --> `uvicorn api.main:app --reload --port 8000`
* **EITHER** open an API platform like Postman and import the curl:
```
curl --location 'http://127.0.0.1:8000/predict/' \
--form 'file=@"/pathToImage.jpg"' \
--form 'model_name="resnet50"'
--form 'nb_similar_images="3"' \
--form 'retrieved_ref="id,score"' \
```

* **OR** use a python request package copying the snippet:

```
import requests
from pathlib import Path
image_path = Path("/pathToImage.jpg")
url = "http://127.0.0.1:8000/predict/"
data = {
        "image_model_name": "resnet50",
        "nb_similar_images": 3,
        "retrieved_ref": "id"
    }
response = requests.post(url, files={"file": open(image_path, "rb")}, data=data)
response.json()
```

**NOTE:** You can find the API documentation on http://127.0.0.1:8000/redoc/ while it is running

#### Gradio-specific

* Run the App --> `python .\app\gradio_app.py` and open the browser to interact with it


## Technical Choices

#### ML models for feature extraction (ResNet50 and VGG16)

* **Choice rationale:** The current version implements two options: ResNet50 (default) and VGG16. Both are CNNs pre-trained on large image datasets. They are well-known for their effectiveness in image classification and feature extraction tasks and are powerful at capturing visual features from images.
* **Trade-offs:** With this choise, I also wanted to balance complexity and to offer the predictions between both a simpler model (VGG16) and a deeper one (ResNet50) given that precision is not a strict requirement. No fine-tuning and Image labelling were done in this regard.

#### Preprocessing pipelines:

* **Choice rationale**: Very basic preprocessing pipeline before feature extraction with no fine-tuning to focus more on the API and app development than the model selection and optimization. The preprocessing steps only involve resizing(`224x224` by default), normalization (with ImageNet mean & std as commonly used : `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`) and data type conversion 
* **Trade-offs:** Preprocessing was not essential to meet the requirements so I simplified this step to not add computational costs or complexity.

#### Similarity measure (Cosine similarity)

* **Choice rationale:** Cosine similarity focuses on the orientation of feature vectors rather than their norms. This is particularly useful for image retrieval where the relative differences in features are more important than their absolute values.

#### Pickle for fature database:

* **Choice rationale**: Pickle is a convenient way to serialize and deserialize Python objects, such as dictionaries of image features.
* **Trade-offs:** Pickle might not be the most efficient for handling large-scale data in deployed environments but was just fine for my current use-case of quick similarity lookup.

#### API Framework:

* **Choice rationale**: FastAPI was selected for its speed, ease of use, and automatic generation of interactive API documentation via Swagger and ReDoc.
* **Trade-offs:** FastAPI is relatively new compared to frameworks like Flask or Django but its speed and ease of use were key concerns for this project.

#### (Optional) Interactive UI with Gradio App

* **Choice rationale:** Gradio provides an intuitive interface to build web-based prototypes without frontend development. It allows users to interact with the system directly from their browser, making the system easily accessible.
